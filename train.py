import sys
import time


import numpy as np
from torch import optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
from utils.path_hyperparameter import ph
import torch
from utils.losses import FCCDN_loss_without_seg
import os
import logging
import random
from models.Models import DPCD
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score,JaccardIndex
from utils.utils import train_val
from utils.dataset_process import compute_mean_std
from utils.dataset_process import image_shuffle, split_image
import onnx
import onnx.utils
import onnx.version_converter
import netron
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def auto_experiment():
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)


def train_net(dataset_name):

    t1_mean, t1_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t1/')
    t2_mean, t2_std = compute_mean_std(images_dir=f'./{dataset_name}/train/t2/')
    dataset_args = dict(t1_mean=t1_mean.tolist(), t1_std=t1_std.tolist(), t2_mean=t2_mean.tolist(),
                        t2_std=t2_std.tolist())

    train_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/train/t1/',
                                 t2_images_dir=f'./{dataset_name}/train/t2/',
                                 labels_dir=f'./{dataset_name}/train/change_label/',
                                 t1_seg_dir=f'./{dataset_name}/train/t1_label/',
                                 t2_seg_dir=f'./{dataset_name}/train/t2_label/',
                                 all_seg_dir=f'./{dataset_name}/train/all_label/',
                                 train=True, **dataset_args)

    val_dataset = BasicDataset(t1_images_dir=f'./{dataset_name}/val/t1/',
                               t2_images_dir=f'./{dataset_name}/val/t2/',
                               labels_dir=f'./{dataset_name}/val/change_label/',
                               t1_seg_dir=f'./{dataset_name}/val/t1_label/',
                               t2_seg_dir=f'./{dataset_name}/val/t2_label/',
                               all_seg_dir=f'./{dataset_name}/val/all_label/',
                               train=False, **dataset_args)

    # 2. Markdown dataset size
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders

    loader_args = dict(num_workers=4,
                       prefetch_factor=5,
                       persistent_workers=True,
                       pin_memory=True,
                       )
    train_loader = DataLoaderX(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoaderX(val_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 4. Initialize logging

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # working device
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime

    logFileLoc = ph.save_dir + ph.logFile
    os.makedirs(os.path.dirname(logFileLoc), exist_ok=True)
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
    logger.write(f'''Starting training:
    Epochs:          {ph.epochs}
    Batch size:      {ph.batch_size}
    Learning rate:   {ph.learning_rate}
    Training size:   {n_train}
    Validation size: {n_val}
    Checkpoints:     {ph.save_checkpoint}
    save best model: {ph.save_best_model}
    Device:          {device.type}
    Mixed Precision: {ph.amp}
    ''')
    logger.flush()
    logger.close()

    # 5. Set up model, optimizer, warm_up_scheduler, learning rate scheduler, loss function and other things

    net = DPCD()  # change detection model
    net = net.to(device=device)
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer
    warmup_lr = np.arange(1e-7, ph.learning_rate,
                          (ph.learning_rate - 1e-7) / ph.warm_up_step)  # warm up learning rate
    grad_scaler = torch.cuda.amp.GradScaler()  # loss scaling for amp

    # load model and optimizer
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate

    criterion = FCCDN_loss_without_seg  # loss function

    best_metrics = dict.fromkeys(['best_f1score', 'lowest loss'], 0)  # best evaluation metrics
    metric_collection = MetricCollection({
        # 'accuracy': Accuracy(task='binary').to(device=device),
        'iou': JaccardIndex(task='binary', num_classes=2).to(device=device),
        'precision': Precision(task='binary').to(device=device),
        'recall': Recall(task='binary').to(device=device),
        'f1score': F1Score(task='binary').to(device=device)
    })  # metrics calculator

    to_pilimg = T.ToPILImage()  # convert to PIL image to log in wandb

    checkpoint_path = rf'.\run\{dataset_name}\{ph.mytime}/'
    best_f1score_model_path = rf'.\run\{dataset_name}\{ph.mytime}/'
    best_loss_model_path = rf'.\run\{dataset_name}\{ph.mytime}/'

    non_improved_epoch = 0
    # 5. Begin training

    for epoch in range(ph.epochs):
        epoch_check=epoch+ph.load_epoch
        net, optimizer, grad_scaler, total_step, lr = \
            train_val(
                mode='train', dataset_name=dataset_name,
                dataloader=train_loader, device=device,
                # log_wandb=log_wandb,
                logFileLoc=logFileLoc,
                net=net,
                optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch_check,
                warmup_lr=warmup_lr, grad_scaler=grad_scaler
            )
        print("train success")
        # 6. Begin evaluation

        # starting validation from evaluate epoch to minimize time
        if epoch_check+1 >= ph.evaluate_epoch:
            with torch.no_grad():
                 net, optimizer, total_step, lr, best_metrics, non_improved_epoch = \
                    train_val(
                        mode='val', dataset_name=dataset_name,
                        dataloader=val_loader, device=device,
                        # log_wandb=log_wandb,
                        logFileLoc=logFileLoc,
                        net=net,
                        optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch_check,
                        best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_f1score_model_path=best_f1score_model_path, best_loss_model_path=best_loss_model_path,
                        non_improved_epoch=non_improved_epoch
                    )
            print(f"val success")




if __name__ == '__main__':

    auto_experiment()

