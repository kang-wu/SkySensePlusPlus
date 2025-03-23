import os
import glob
import numpy as np
import yaml
import argparse
import oss2
import torch
from PIL import Image
from tqdm import tqdm
import concurrent.futures
from torchvision.transforms import functional as F
import random


from antmmf.common.registry import registry
from antmmf.common.report import Report, default_result_formater
from antmmf.structures import Sample, SampleList
from antmmf.predictors.base_predictor import BasePredictor
from antmmf.utils.timer import Timer
from antmmf.predictors.build import build_predictor
from antmmf.common.task_loader import build_collate_fn
from antmmf.datasets.samplers import SequentialSampler
from antmmf.common.build import build_config

from lib.utils.checkpoint import SegCheckpoint
from lib.datasets.loader.few_shot_flood3i_loader import FewShotFloodLoader


def seed_everything(seed=0):
    # 为了确保CUDA卷积的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

@registry.register_predictor("OneshotPredictor")
class OneshotPredictor(BasePredictor, FewShotFloodLoader):
    def __init__(self, config):
        self.config = config
        self.predictor_parameters = self.config.predictor_parameters

    def _predict(self, sample_list):
        # with torch.no_grad():
        if True:
            sample_list = sample_list.to(self.device)
            report = self._forward_pass(sample_list)
        return report, sample_list

    def _forward_pass(self, samplelist):
        autocast_dtype = torch.bfloat16
        with torch.cuda.amp.autocast(enabled=True, dtype=autocast_dtype): 
            model_output = self.model(samplelist)
        report = Report(samplelist, model_output)
        return report

    def predict(self, data=None):
        if data is None:
            data = self.dummy_request()
        sample = self._build_sample(data)
        if not isinstance(sample, Sample):
            raise Exception(
                f"Method _build_sample is expected to return a instance of antmmf.structures.sample.Sample,"
                f"but got type {type(sample)} instead.")
        result, sample_list = self._predict(SampleList([sample]))
        np_result = default_result_formater(result)
        result = self.format_result(np_result)
        assert isinstance(
            result, dict
        ), f"Result should be instance of Dict,but got f{type(result)} instead"
        return result, sample_list

    def load_checkpoint(self):
        self.resume_file = self.config.predictor_parameters.model_dir
        self.checkpoint = SegCheckpoint(self, load_only=True)
        self.checkpoint.load_model_weights(self.resume_file, force=True)

    def covert_speedup_op(self):
        if self.config.predictor_parameters.replace_speedup_op:
            from lib.utils.optim_utils import replace_speedup_op
            self.model = replace_speedup_op(self.model)

def save_image(output_path, image_np_array):
    image = Image.fromarray(image_np_array)
    image.save(output_path)

def build_predictor_from_args(args, *rest, **kwargs):
    config = build_config(
        args.config,
        config_override=args.config_override,
        opts_override=args.opts,
        specific_override=args,
    )
    predictor_obj = build_predictor(config)
    setattr(predictor_obj, "args", args)
    return predictor_obj


def build_online_predictor(model_dir=None, config_yaml=None):
    assert model_dir or config_yaml
    from antmmf.utils.flags import flags

    # if config_yaml not indicated, there must be a `config.yaml` file under `model_dir`
    config_path = config_yaml if config_yaml else os.path.join(model_dir, "config.yaml")
    input_args = ["--config", config_path]
    if model_dir is not None:
        input_args += ["predictor_parameters.model_dir", model_dir]
    parser = flags.get_parser()
    args = parser.parse_args(input_args)
    predictor = build_predictor_from_args(args)
    return predictor

def profile(profiler, text):
    print(f'{text}: {profiler.get_time_since_start()}')
    profiler.reset()

def cvt_colors(img_2d, idx_2_color_rgb):
    img_rgb = np.zeros((img_2d.shape[0], img_2d.shape[1], 3), dtype=np.uint8)
    for idx, color in idx_2_color_rgb.items():
        img_rgb[img_2d==idx] = color
    return img_rgb

def process_results(preds, targets, input_imgs, img_names, save_dir, save_dir_vis, idx_2_color):
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    idx_2_color_rgb = {}
    
    for idx, color in idx_2_color.items():
        r = color // (256 * 256)
        g = (color % (256 * 256)) // 256
        b = color % 256
        idx_2_color_rgb[idx] = (r, g, b)
    
    for i in range(preds.size(0)):
        output1 = preds[i].argmax(0) # h, w
        output1_total = output1.clone()
        output1 = output1[output1.shape[0]//2:, :]
        output1 = output1.numpy().astype(np.uint8)
        output1 = cvt_colors(output1, idx_2_color_rgb)

        output1_total = output1_total.numpy().astype(np.uint8)
        output1_total = cvt_colors(output1_total, idx_2_color_rgb)

        # for visualization
        output2 = targets[i] 
        output2 = output2.numpy().astype(np.uint8)
        output2 = cvt_colors(output2, idx_2_color_rgb)

        input_img = torch.einsum('chw->hwc', input_imgs[i])
        input_img = torch.clip((input_img * imagenet_std + imagenet_mean) * 255, 0, 255)
        input_img = input_img.numpy().astype(np.uint8)

        output_comb = np.concatenate((input_img, output1_total, output2), axis=1)
        
        # save result
        save_path = os.path.join(save_dir, f'{img_names[i]}.png')
        save_image(save_path, output1)
        save_path_vis = os.path.join(save_dir_vis, f'{img_names[i]}.png')
        save_image(save_path_vis, output_comb)

def test(args):
    model_path = args.model_path
    config_path = args.config
    global_seed = args.seed
    predictor = build_online_predictor(model_path, config_path)
    seed_everything(global_seed)

    dataset = FewShotFloodLoader(
        "test", predictor.config.task_attributes.segmentation.dataset_attributes.few_shot_flood_segmentation)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        sampler=SequentialSampler(dataset),
        collate_fn=build_collate_fn(dataset),
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )

    print(len(loader))

    predictor.load(with_ckpt=True)

    predictor.covert_speedup_op()

    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_vis = os.path.join(args.save_dir, 'vis_full')
    if not os.path.exists(save_dir_vis):
        os.makedirs(save_dir_vis)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        profiler2 = Timer()
        profiler2.reset()
        for sample_batched in tqdm(loader):
            profile(profiler2, "Build sample time")
            result, sample_list = predictor._predict(SampleList(sample_batched))
            profile(profiler2, "Infer time")
            preds = result["logits_hr"].to(torch.float32).detach().cpu()
            targets = result['mapped_targets'].to(torch.float32).detach().cpu()
            idx_2_color = result['idx_2_color']
            input_imgs = sample_list['hr_img'].to(torch.float32).detach().cpu()
            img_names = sample_list["img_name"]

            executor.submit(process_results, preds, targets, input_imgs, img_names, save_dir, save_dir_vis, idx_2_color)
            profile(profiler2, "Save results time")
        try:
            del predictor.model
        except Exception as e:
            print('delete model error: ', e)

def parse_args():
    desc = '1-shot predictor'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_path',
                        required=True,
                        type=str,
                        help='model directory')
    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help='seed')
    parser.add_argument('--config',
                        required=True,
                        type=str,
                        help='config path')
    parser.add_argument('--save_dir',
                        required=False,
                        type=str,
                        help='save directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)
