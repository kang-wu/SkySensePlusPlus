# Copyright (c) Ant Group. and its affiliates.
import gc
import math
from itertools import chain

import torch
from torch import nn
from tqdm import tqdm

from antmmf.common.registry import registry
from antmmf.common.report import Report
from antmmf.common.meter import Meter
from antmmf.modules.metrics import Metrics
from antmmf.optimizer.combine_optimizers import CombinedOptimizer
from antmmf.utils.distributed_utils import (broadcast_scalar, is_main_process)
from antmmf.utils.early_stopping import EarlyStopping
from antmmf.utils.general import clip_gradients, count_parameters, nullcontext
from antmmf.utils.timer import Timer
from antmmf.trainers.base_trainer import BaseTrainer

from lib.utils.utils import cancel_gradients_backbone, EMA
from lib.utils.checkpoint import SegCheckpoint

try:
    import atorch
    from atorch import amp
except ImportError:
    pass


@registry.register_trainer("seg_trainer")
class SEGTrainer(BaseTrainer):

    def __init__(self, config):
        super().__init__(config)
        self.enable_torch_amp=True
        self.enable_atorch_amp=False

    def load(self, has_check_point=True):
        super().load(has_check_point)
        torch.backends.cuda.matmul.allow_tf32 = self.config.training_parameters.get(
            "enable_tf32", False)
        if hasattr(
                self.config.training_parameters, "freeze_backbone"
        ) and self.config.training_parameters.freeze_backbone is True:
            for n, p in self.model.named_parameters():
                if "backbone_hr." in n or 'backbone_s2.' in n or 'head_s2.' in n or 'backbone_s1.' in n or 'head_s1.' in n or 'fusion.' in n or 'ctpe' in n or 'glbank.' in n:
                    p.requires_grad = False
                else:
                    print(n, '-->', p.requires_grad)
        if hasattr(self.config.training_parameters,
                   "ema") and self.config.training_parameters.ema is True:
            self.ema = EMA(self.model, 0.96)
            self.ema.register()

    def load_extras(self, has_check_point=True):
        self.checkpoint = None if has_check_point is False else SegCheckpoint(
            self)
        self.meter = Meter()

        self.training_parameters = self.config.training_parameters

        monitored_metric = self.training_parameters.monitored_metric
        metric_minimize = self.training_parameters.metric_minimize
        should_early_stop = self.training_parameters.should_early_stop
        patience = self.training_parameters.patience

        self.log_interval = self.training_parameters.log_interval
        self.snapshot_interval = self.training_parameters.snapshot_interval
        self.max_iterations = self.training_parameters.max_iterations
        self.should_clip_gradients = self.training_parameters.clip_gradients
        self.max_epochs = self.training_parameters.max_epochs
        self.gradient_accumulation_steps = int(
            self.training_parameters.gradient_accumulation_steps)
        assert self.gradient_accumulation_steps >= 1
        for t_type in self.task_loader.task_type:
            if t_type == "train":
                self.dataset_train_order = self.training_parameters.get(
                    "dataset_train_order", self.train_task.datasets_name)
            if t_type == "val":
                self.dataset_val_order = self.training_parameters.get(
                    "dataset_val_order", self.val_task.datasets_name)
            if t_type == "test":
                self.dataset_test_order = self.training_parameters.get(
                    "dataset_test_order", self.test_task.datasets_name)
            if t_type == "interpret":
                self.dataset_interpret_order = self.training_parameters.get(
                    "dataset_interpret_order",
                    self.interpret_task.datasets_name)

        self.early_stopping = EarlyStopping(
            self.model,
            self.checkpoint,
            monitored_metric,
            patience=patience,
            minimize=metric_minimize,
            should_stop=should_early_stop,
        )
        self.current_epoch = 1
        self.current_iteration = 0

        self.not_debug = self.training_parameters.logger_level != "debug"

        self.lr_scheduler = None
        self.setup_lr_scheduler()

        if self.checkpoint is not None:
            self.checkpoint.load_state_dict()

        if "overall_metrics" in self.training_parameters:
            self.overall_metric_evaluator = Metrics(
                self.config.training_parameters.get("overall_metrics", []))
        self.synchronized_loss = self.config.training_parameters.synchronized_loss

    def train(self):
        self.writer.write("===== Model =====")
        self.writer.write(self.model)
        self.writer.write(
            "Model Params: Trainable {Trainable:.3f}M  Total {Total:.3f}M".
            format(**count_parameters(self.model)))

        if "train" not in self.run_type:
            self.inference()
            return

        should_break = False

        if self.max_epochs is None:
            self.max_epochs = math.inf
        else:
            self.max_iterations = min(self.max_iterations,
                                      self.max_epochs * self.epoch_iterations)

        self.model.train()
        self.train_timer = Timer()

        self.profile("Setup Time")

        if self.enable_torch_amp:
            self.writer.write("Using Automatic mixed precision training")
            if hasattr(self.config, "amp_attributes") and hasattr(
                    self.config.amp_attributes, "growth_interval"):
                growth_interval = self.config.amp_attributes.growth_interval
            else:
                growth_interval = 2000
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=self.config.amp_attributes.init_scale,
                enabled=False,
                growth_interval=growth_interval)
            self.writer.write("Using Init scale:%s" %
                              self.config.amp_attributes.init_scale)

        self.optimizer.zero_grad()

        self.writer.write("Starting training...")
        while self.current_iteration < self.max_iterations and not should_break:
            registry.register("current_epoch", self.current_epoch)
            self.task_loader.seed_sampler("train", self.current_epoch)

            if self.current_epoch > self.max_epochs:
                break

            for batch in tqdm(
                    chain(*self.train_loader_list),
                    total=self._len_of_loader_list(self.train_loader_list),
                    disable=self.disable_tqdm or (not is_main_process()),
            ):
                self.profile("Batch load time")
                report, _, _ = self._forward_pass(
                    batch, enable_amp=self.enable_torch_amp)
                if report is None:
                    continue

                self._update_meter(report, self.meter)

                loss = self._extract_loss(report)
                self._backward(loss)
                if hasattr(
                        self.config.training_parameters,
                        "ema") and self.config.training_parameters.ema is True:
                    self.ema.update()
                should_break = self._logistics()

                self._run_scheduler()

                self.current_iteration += 1
                self.writer.write(self.current_iteration, "debug")
                registry.register("current_iteration", self.current_iteration)
                if self.current_iteration >= self.max_iterations:
                    break
                if should_break:
                    break

            self.current_epoch += 1

        self.finalize()

    def _forward_pass(self, batch, enable_amp=False):
        if not batch:  # Samplelist might be empty dict
            return None, None, None
        prepared_batch = self.task_loader.prepare_batch(batch)

        self.profile("Batch prepare time")
        forward_context = torch.cuda.amp.autocast(
            enabled=True,
            dtype=torch.bfloat16) if enable_amp else nullcontext()

        with forward_context:
            # Arguments should be a dict at this point
            model_output = self.model(prepared_batch)

            if self.synchronized_loss:
                is_parallel = isinstance(
                    self.model, nn.DataParallel) or isinstance(
                        self.model, nn.parallel.DistributedDataParallel)
                if "losses" not in model_output:
                    loss_func = getattr(
                        self.model.module if is_parallel else self.model,
                        "losses")
                    model_output["losses"] = loss_func(
                        prepared_batch,
                        model_output,
                        iteration=self.current_iteration)
                if "metrics" not in model_output:
                    metric_func = getattr(
                        self.model.module if is_parallel else self.model,
                        "metrics")
                    model_output["metrics"] = metric_func(
                        prepared_batch, model_output)

        report = Report(prepared_batch, model_output)
        self.profile("Forward time")

        return report, model_output, prepared_batch

    def _backward(self, loss):
        loss = loss / self.gradient_accumulation_steps

        if self.enable_torch_amp:
            self.scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place, this should
            # be called first so that clip_gradients can take effect as usual.
            self.scaler.unscale_(self.optimizer)
        elif self.enable_atorch_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.profile("Backward time")

        if self.current_iteration % self.gradient_accumulation_steps != 0:
            return

        if self.should_clip_gradients:
            if self.enable_atorch_amp:
                clip_gradients(amp.master_params(self.optimizer),
                               self.current_iteration, self.writer,
                               self.config)
            else:
                clip_gradients(self.model, self.current_iteration, self.writer,
                               self.config)

        if hasattr(
                self.config.training_parameters, "freeze_backbone_steps"
        ) and self.config.training_parameters.freeze_backbone_steps is not None:
            cancel_gradients_backbone(
                self.current_iteration, self.model,
                self.config.training_parameters.freeze_backbone_steps)

        if self.enable_torch_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.profile("Optimizer time")

    def _logistics(self):
        should_print = self.current_iteration and self.current_iteration % self.log_interval == 0
        extra = {}
        prefix = ""

        if should_print is True:
            if "cuda" in str(self.device):
                extra["max mem"] = torch.cuda.max_memory_allocated() / 1024
                extra["max mem"] //= 1024

            # display lr
            if isinstance(self.optimizer, CombinedOptimizer):
                extra["lr"] = self.optimizer.get_optimizers_lr_str()
            else:
                extra["lr"] = "|".join([
                    "{:.8f}".format(x["lr"]).rstrip("0")
                    for x in self.optimizer.param_groups
                ])

            extra.update({
                "time": self.train_timer.get_time_since_start(),
                "eta": self._calculate_time_left(),
            })

            self.train_timer.reset()

        self._summarize_meter(
            self.meter,
            prefix=prefix,
            extra=extra,
            should_print=should_print,
        )

        should_break = self._try_full_validation()

        return should_break

    def _try_full_validation(self, force=False):
        should_break = False

        if self.current_iteration and self.current_iteration % self.snapshot_interval == 0 or force:
            self.writer.write(
                "Evaluation time. Running on full validation set...")

            validation_timer = Timer()
            dataset_name, meter = self.evaluate_set(self.val_loader_list)
            extra = {
                "validation time": validation_timer.get_time_since_start()
            }

            overall_metric = self.overall_metric_evaluator.summarize()
            stop = self.early_stopping(self.current_iteration, overall_metric,
                                       meter)
            if hasattr(self.config.training_parameters,
                       "ema") and self.config.training_parameters.ema is True:
                self.ema.restore()
            stop = bool(broadcast_scalar(stop, src=0, device=self.device))

            extra.update(self.early_stopping.get_info())

            prefix = "{}: full val".format(dataset_name)
            self._summarize_overall(overall_metric,
                                    meter,
                                    prefix=prefix,
                                    extra=extra)
            gc.collect()

            if "cuda" in str(self.device):
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

            if stop > 0:  # `stop` is now `int`, NCCL does not support `boolean` type's broadcasting
                self.writer.write("Early stopping activated")
                should_break = True

        return should_break

    def evaluate_set(self, loader_list):
        from antmmf.structures import SampleList

        meter = Meter()
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.model.eval()
            if hasattr(self.config.training_parameters,
                       "ema") and self.config.training_parameters.ema is True:
                self.ema.apply_shadow()
            if self.config.training_parameters.get('fp16', False):
                self.model.half()
            self.overall_metric_evaluator.reset()
            for idx, batch in tqdm(
                    enumerate(chain(*loader_list)),
                    total=self._len_of_loader_list(loader_list),
                    disable=not is_main_process() or self.disable_tqdm,
            ):
                # report, model_output, prepared_batch = self._forward_pass(
                #     batch, enable_amp=self.enable_torch_amp)
                if idx >= self.config.training_parameters.get('num_eval', 1e7):
                    break
                if self.config.training_parameters.get('fp16', False):
                    input_dict = SampleList()
                    for k, v in batch.items():
                        if isinstance(v, torch.cuda.FloatTensor):
                            input_dict[k] = v.half()
                        else:
                            input_dict[k] = v
                    report, model_output, prepared_batch = self._forward_pass(
                        input_dict, enable_amp=self.enable_torch_amp)
                else:
                    report, model_output, prepared_batch = self._forward_pass(
                        batch, enable_amp=self.enable_torch_amp)
                self._update_meter(report, meter)
                self.overall_metric_evaluator.collect(prepared_batch,
                                                      model_output)
            for _, metric_object in self.overall_metric_evaluator.metrics.items(
            ):
                metric_object.all_reduce()
            self.model.train()

        return report.dataset_name, meter