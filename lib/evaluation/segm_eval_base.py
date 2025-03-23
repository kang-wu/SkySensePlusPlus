import os
import argparse

import numpy as np
import pandas as pd
from skimage.transform import resize

from skimage import io
from multiprocessing import Pool
from functools import partial
import logging
import datetime

def get_confusion_matrix(pred, gt, num_class):
    assert pred.shape == gt.shape,  f"pred.shape: {pred.shape} != gt.shape: {gt.shape}"
    mask = (gt >= 0) & (gt < num_class) # 去掉为0的背景类别
    label = num_class * gt[mask] + pred[mask]
    count = np.bincount(label, minlength=num_class**2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix

def get_miou(confusion_matrix):
    diagonal_elements = np.diag(confusion_matrix)
    column_sums = np.sum(confusion_matrix, axis=0)
    row_sums = np.sum(confusion_matrix, axis=1)
    ious = diagonal_elements/(column_sums + row_sums - diagonal_elements)
    m_iou = np.nanmean(ious)
    return m_iou

def get_mprecison(confusion_matrix):
    diagonal_elements = np.diag(confusion_matrix)
    column_sums = np.sum(confusion_matrix, axis=0)
    precisions = diagonal_elements / (column_sums + 1e-06) 
    m_precision = np.nanmean(precisions)
    return m_precision

def get_mrecall(confusion_matrix):
    diagonal_elements = np.diag(confusion_matrix)
    row_sums = np.sum(confusion_matrix, axis=1)
    recalls= diagonal_elements / (row_sums + 1e-06)
    m_recall = np.nanmean(recalls)
    return m_recall 

def get_macc(confusion_matrix):
    '''
    acc = tp/tp+fn 就是recall
    '''
    m_recall = get_mrecall(confusion_matrix)
    return m_recall 

def get_per_class_iou(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - intersection
    iou = intersection / (union.astype(np.float32) + 1e-6)
    return iou

def get_per_class_acc(confusion_matrix):
    total_acc = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1).astype(np.float32) + 1e-6)
    return total_acc


def post_process_segm_output(segm, colors, dist_type='abs'):
    """
    Post-processing to turn output segm image to class index map using NumPy
    Args:
        segm: (H, W, 3)
    Returns:
        class_map: (H, W)
    """
    palette = np.array(colors)
    segm = segm.astype(np.float32)  # (h, w, 3)
    h, w, k = segm.shape[0], segm.shape[1], palette.shape[0]
    if dist_type == 'abs':
        dist = np.abs(segm.reshape(h, w, 1, 3) - palette.reshape(1, 1, k, 3))  # (h, w, k)
    elif dist_type == 'square':
        dist = np.power(segm.reshape(h, w, 1, 3) - palette.reshape(1, 1, k, 3), 2)  # (h, w, k)
    elif dist_type == 'mean':
        dist_abs = np.abs(segm.reshape(h, w, 1, 3) - palette.reshape(1, 1, k, 3))  # (h, w, k)
        dist_square = np.power(segm.reshape(h, w, 1, 3) - palette.reshape(1, 1, k, 3), 2)  # (h, w, k)
        dist = (dist_abs + dist_square) / 2.
    else:
        raise NotImplementedError
    
    dist = np.sum(dist, axis=-1)
    pred = np.argmin(dist, axis=-1).astype(np.int)
    return pred
    
def get_args_parser():
    parser = argparse.ArgumentParser('semantic segmentation evaluation', add_help=False)
    parser.add_argument('--pred_dir', type=str, help='dir to pred', required=True)
    parser.add_argument('--gt_dir', type=str, help='dir to gt', required=True)
    parser.add_argument('--gt_list_path', type=str, help='dir to gt_list_path', required=True)
    parser.add_argument('--gt_suffix', type=str, help='suffix to gt', required=True)
    parser.add_argument('--dataset_name', type=str, help='dataset name', required=True)
    parser.add_argument('--model_name', type=str, help='model name', required=True)
    parser.add_argument('--dist_type', type=str, help='dist type',
                        default='abs', choices=['abs', 'square', 'mean'])
    return parser.parse_args()

def process_file(file_dict, pred_dir, gt_dir, args, num_class):
    filename = file_dict['file_name']
    file_cls = file_dict['file_cls']

    gt = io.imread(os.path.join(gt_dir, filename))
    gt_index = gt.copy()
    gt_index[gt_index != file_cls] = 0
    gt_index[gt_index == file_cls] = 1

    try:
        pred = io.imread(os.path.join(pred_dir, filename.replace('.png', f'-{file_cls}.png')))
        pred = resize(pred, gt.shape[-2:], anti_aliasing=False, mode='reflect', order=0)
        
        if len(pred.shape) == 3:
            pred_index = pred[:,:,0].copy()
        else:
            pred_index = pred.copy()
        pred_index[pred_index<=127] = 0
        pred_index[pred_index>127] = 1
    except:
        logging.info(filename.replace('.png', f'_{file_cls}.png'), 'not found!')
        pred_index = gt_index.copy()
    
    pred_index = pred_index.flatten()
    gt_index = gt_index.flatten()
    confusion_matrix = get_confusion_matrix(pred_index, gt_index, num_class)
    return file_cls, confusion_matrix

if __name__ == '__main__':
    args = get_args_parser()
    dataset_name = args.dataset_name 
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    gt_list_path = args.gt_list_path
    dist_type = args.dist_type
    model_name = args.model_name

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs('logs/eval', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/eval/eval_{model_name}_{dataset_name}_{current_time}.log'),
            logging.StreamHandler() 
        ]
    )

    output_folder = os.path.join(pred_dir, f'eval_{dataset_name}')
    os.makedirs(output_folder, exist_ok=True)


    num_class = 2

    with open(gt_list_path, 'r') as f:
        file_cls_list = f.readlines()
    
    file_list = []
    for i in file_cls_list:
        i = i.strip()
        file_name = i[:-3]
        file_cls = i[-2:]
        file_list.append({'file_name': file_name, 'file_cls': int(file_cls)})
    
    all_pred_labels = []
    all_gt_labels = []

    process_file_partial = partial(process_file, pred_dir=pred_dir,gt_dir=gt_dir, args=args, num_class=num_class)
    
    pool = Pool()

    outputs = pool.map(process_file_partial, file_list)
    
    pool.close()
    pool.join()
    logging.info(f'len outputs: {len(outputs)}')
    confusion_matrix_dict = {}
    for cls, confusion_matrix in outputs:
        if cls in confusion_matrix_dict.keys():
            confusion_matrix_dict[cls] += confusion_matrix
        else:
            confusion_matrix_dict[cls] = confusion_matrix

    class_list = []
    iou_list = []
    acc_list = []
    for cls, confusion_matrix in confusion_matrix_dict.items():
        ious = get_per_class_iou(confusion_matrix)
        accs = get_per_class_acc(confusion_matrix)
        logging.info(f'cls: {cls}, ious: {ious}, accs: {accs}')
        class_list.append(cls)
        iou_list.append(ious[1])
        acc_list.append(accs[1])
    
    miou = np.mean(iou_list)
    macc = np.mean(acc_list)

    df_metrics = pd.DataFrame({
        'Class': class_list + ['Mean'],
        'IoU': iou_list + [miou],
        'Accuracy':  acc_list + [macc],
    })
    pd.set_option('display.float_format', '{:.4f}%'.format)
    logging.info(df_metrics)
    pd.reset_option('display.float_format')
    df_metrics.to_csv(os.path.join(output_folder, 'eval.csv'), index=False, float_format='%.4f')
    

