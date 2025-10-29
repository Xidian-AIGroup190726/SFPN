import sys
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
import logging
from utils.path_hyperparameter import ph
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score,JaccardIndex
from models.Models import DPCD
from utils.dataset_process import compute_mean_std
from tqdm import tqdm
# import bpd_cd_cuda_3
import math

#边缘
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist

def boundary_f1_score(pred, gt, tolerance=10):
    """
    Boundary F1-score
    pred, gt: 2D numpy arrays (0/1)
    tolerance: pixel distance tolerance for matching boundaries
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_boundary = find_boundaries(pred, mode='outer').astype(np.uint8)
    gt_boundary = find_boundaries(gt, mode='outer').astype(np.uint8)

    pred_coords = np.argwhere(pred_boundary)
    gt_coords = np.argwhere(gt_boundary)

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 0.0

    dist_matrix = cdist(pred_coords, gt_coords)

    matched_pred = np.any(dist_matrix <= tolerance, axis=1).sum()
    matched_gt = np.any(dist_matrix <= tolerance, axis=0).sum()

    precision = matched_pred / len(pred_coords)
    recall = matched_gt / len(gt_coords)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def boundary_iou(pred, gt, boundary_width=10):
    """
    Boundary IoU
    pred, gt: 2D numpy arrays (0/1)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_boundary = find_boundaries(pred, mode='outer')
    gt_boundary = find_boundaries(gt, mode='outer')

    pred_dil = binary_dilation(pred_boundary, iterations=boundary_width)
    gt_dil = binary_dilation(gt_boundary, iterations=boundary_width)

    intersection = np.logical_and(pred_dil, gt_dil).sum()
    union = np.logical_or(pred_dil, gt_dil).sum()

    return intersection / union if union > 0 else 0.0

def train_net(dataset_name, load_checkpoint=True):
    # 1. Create dataset

    # compute mean and std of train dataset to normalize train/val/test dataset
    t1_mean, t1_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t1/')
    t2_mean, t2_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t2/')

    dataset_args = dict(t1_mean=t1_mean.tolist(), t1_std=t1_std.tolist(), t2_mean=t2_mean.tolist(),
                        t2_std=t2_std.tolist())

    test_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/test/t1/',
                               t2_images_dir=f'./{dataset_name}/test/t2/',
                               labels_dir=f'./{dataset_name}/test/change_label/',
                               t1_seg_dir=f'./{dataset_name}/test/t1_label/',
                               t2_seg_dir=f'./{dataset_name}/test/t2_label/',
                               all_seg_dir=f'./{dataset_name}/test/all_label/',
                               train=False, **dataset_args)
    # 2. Create data loaders
    loader_args = dict(num_workers=4,
                       prefetch_factor=5,
                       persistent_workers=True
                       )
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 3. Initialize logging
    logging.basicConfig(level=logging.INFO)

    # 4. Set up device, model, metric calculator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Using device {device}')
    net = DPCD()
    net.to(device=device)

    assert ph.load, 'Loading model error, checkpoint ph.load'
    load_model = torch.load(ph.load, map_location=device)
    if load_checkpoint:
        net.load_state_dict(load_model['net'])
    else:
        net.load_state_dict(load_model)
    logging.info(f'Model loaded from {ph.load}')
    torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')

    metric_collection = MetricCollection({
        'iou': JaccardIndex(task='binary', num_classes=2).to(device=device),
        'precision': Precision(task='binary').to(device=device),
        'recall': Recall(task='binary').to(device=device),
        'f1score': F1Score(task='binary').to(device=device)
    })  # metrics calculator

    net.eval()
    logging.info('SET model mode to test!')
    test_path=r'./test_result/'
    with torch.no_grad():
        for batch_img1, batch_img2, labels,label_t1_seg,label_t2_seg,label_all_seg,weight_t1,weight_t2,weight_all,name in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.float().to(device)
            label_t1_seg = label_t1_seg.float().to(device)
            label_t2_seg = label_t2_seg.float().to(device)
            label_all_seg = label_all_seg.float().to(device)

            pred_change,pred_seg_1 , pred_seg_2 ,pred_seg_all = net(batch_img1, batch_img2)


            batch_size = pred_seg_all.size(0)

            for j in range(batch_size):

                pred_change=pred_change.squeeze(1)
                metric_collection.update(pred_change[j,...], labels[j,...])

                pred_mask = (pred_change[j, ...].cpu().numpy() >= 0.5).astype(np.uint8)
                gt_mask = labels[j, ...].cpu().numpy().astype(np.uint8)

                bf1_val = boundary_f1_score(pred_mask, gt_mask, tolerance=10)
                biou_val = boundary_iou(pred_mask, gt_mask, boundary_width=10)

                if 'bf1_list' not in locals():
                    bf1_list, biou_list = [], []
                bf1_list.append(bf1_val)
                biou_list.append(biou_val)


                cd_preds_array = pred_change[j,...].cpu().numpy()
                cd_preds_image = (cd_preds_array * 255).astype(np.uint8)
                image = Image.fromarray(cd_preds_image, mode='L')
                output_path = test_path+name[j]+'.png'
                image.save(output_path)

                color_map = {
                    'FP': [255, 0, 0],
                    'FN': [0, 0, 255],
                    'TN': [0, 0, 0],
                    'TP': [255, 255, 255]
                }

                pred_change = pred_change.squeeze(1)
                pred_change_array = pred_change[j, ...].cpu().numpy()
                labels_array = labels[j, ...].cpu().numpy()

                # 创建一个空的 RGB 图像数组
                colored_image = np.zeros((pred_change_array.shape[0], pred_change_array.shape[1], 3), dtype=np.uint8)

                fp_mask = (pred_change_array >= 0.5) & (labels_array == 0)
                fn_mask = (pred_change_array < 0.5) & (labels_array == 1)
                tn_mask = (pred_change_array <0.5) & (labels_array == 0)
                tp_mask = (pred_change_array >= 0.5) & (labels_array == 1)

                colored_image[fp_mask] = color_map['FP']
                colored_image[fn_mask] = color_map['FN']
                colored_image[tn_mask] = color_map['TN']
                colored_image[tp_mask] = color_map['TP']

                image = Image.fromarray(colored_image, mode='RGB')  # 'RGB' 表示彩色图像

                output_path = f"{test_path}{name[j]}.png"
                image.save(output_path)

            del batch_img1, batch_img2, labels,label_t1_seg,label_t2_seg,label_all_seg,weight_t1,weight_t2,weight_all,image

        test_metrics = metric_collection.compute()
        print(f"Metrics on all data: {test_metrics}")
        metric_collection.reset()

    print('over')


if __name__ == '__main__':

    try:
        train_net(dataset_name='whu', load_checkpoint=False)
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)
