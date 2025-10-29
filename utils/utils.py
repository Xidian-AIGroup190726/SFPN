from pathlib import Path
import time
import numpy as np
import torch.nn.functional as F
from utils.path_hyperparameter import ph
import torch
from tqdm import tqdm
from PIL import Image
import os

os.environ["ALBUMENTATIONS_SKIP_VERSION_CHECK"] = "1"

def save_model(model, path, epoch, mode, optimizer=None):
    # Ensure mode is one of the expected values
    assert mode in ['checkpoint', 'loss', 'f1score'], "mode should be 'checkpoint', 'loss', or 'f1score'"

    # Create directory if it does not exist
    Path(path).mkdir(parents=True, exist_ok=True)

    # Get current local time
    localtime = time.asctime(time.localtime(time.time()))

    # Construct file path using os.path.join for cross-platform compatibility
    if mode == 'checkpoint':
        filepath = os.path.join(path, rf'checkpoint_epoch{epoch}.pth')
        state_dict = {'net': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None}
    else:
        filepath = os.path.join(path, rf'best_{mode}_epoch{epoch}.pth')
        state_dict = model.state_dict()

    try:
        # Save the model state dict
        torch.save(state_dict, filepath)
        print(f'best {mode} model {epoch} saved at {localtime}!')
    except FileNotFoundError:
        print(f"Directory not found: {filepath}")
    except PermissionError:
        print(f"Permission denied: {filepath}")
    except Exception as e:
        print(f"Failed to save model due to error: {e}")

def train_val(
        mode, dataset_name,
        dataloader, device,logFileLoc,net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_f1score_model_path=None, best_loss_model_path=None, non_improved_epoch=None
):
    assert mode in ['train', 'val'], 'mode should be train, val'
    epoch_loss = 0
    # Begin Training/Evaluating
    if mode == 'train':
        net.train()
    else:
        net.eval()
    batch_iter = 0
    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=n_iter)
    if epoch==0:
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'模型共有 {total_params} 个可训练参数')
    for i, (batch_img1, batch_img2, labels,label_t1_seg,label_t2_seg,label_all_seg,weight_t1,weight_t2,weight_all,name) in enumerate(tbar):
        tbar.set_description(
            "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + ph.batch_size))
        batch_iter = batch_iter + ph.batch_size
        total_step += 1
        if mode == 'train':
            optimizer.zero_grad()
            if total_step < ph.warm_up_step:
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr[total_step]

        batch_img1 = batch_img1.float().to(device)
        batch_img2 = batch_img2.float().to(device)
        labels = labels.float().to(device)
        label_t1_seg=label_t1_seg.float().to(device)
        label_t2_seg=label_t2_seg.float().to(device)
        label_all_seg=label_all_seg.float().to(device)
        weight_t1=weight_t1.float().to(device)
        weight_t2=weight_t2.float().to(device)
        weight_all=weight_all.float().to(device)

        if mode == 'train':
            # using amp
            with torch.cuda.amp.autocast():
                pred_change, pred_seg_1, pred_seg_2 ,pred_seg_all = net(batch_img1, batch_img2)
                loss = criterion(pred_change,pred_seg_1,pred_seg_2,pred_seg_all,labels,label_t1_seg,label_t2_seg,label_all_seg,weight_t1,weight_t2,weight_all)

            change_loss=loss[0]
            seg_loss=sum(loss[1:])
            cd_loss = sum(loss)

            grad_scaler.scale(cd_loss).backward()


            torch.nn.utils.clip_grad_norm_(net.parameters(), 20, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            pred_change, pred_seg_1, pred_seg_2, pred_seg_all = net(batch_img1, batch_img2)
            loss = criterion(pred_change, pred_seg_1, pred_seg_2, pred_seg_all, labels, label_t1_seg, label_t2_seg,label_all_seg,weight_t1,weight_t2,weight_all)
            cd_loss = sum(loss)

        epoch_loss += cd_loss
        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=batch_img1.shape[0])
            # ipdb.set_trace()
            t1_images_dir = Path(f'./{dataset_name}/{mode}/t1/')
            t2_images_dir = Path(f'./{dataset_name}/{mode}/t2/')
            labels_dir = Path(f'./{dataset_name}/{mode}/change_label/')
            t1_seg_dir = Path(f'./{dataset_name}/{mode}/t1_label/')
            t2_seg_dir = Path(f'./{dataset_name}/{mode}/t2_label/')
            all_seg_dir = Path(f'./{dataset_name}/{mode}/change_label/')


            t1_img_log = Image.open(list(t1_images_dir.glob(name[sample_index] + '.*'))[0])
            t2_img_log = Image.open(list(t2_images_dir.glob(name[sample_index] + '.*'))[0])
            label_log = Image.open(list(labels_dir.glob(name[sample_index] + '.*'))[0])
            pred_log = torch.round(pred_change[sample_index]).cpu().clone().float()
            t1_seg_log = Image.open(list(t1_seg_dir.glob(name[sample_index] + '.*'))[0])
            t2_seg_log = Image.open(list(t2_seg_dir.glob(name[sample_index] + '.*'))[0])
            all_seg_log = Image.open(list(all_seg_dir.glob(name[sample_index] + '.*'))[0])
            name_log=name[sample_index]
            # pred_log[pred_log >= 0.5] = 1
            # pred_log[pred_log < 0.5] = 0
            # pred_log = pred_log.float()

        pred_change = pred_change.float()
        labels = labels.int().unsqueeze(1)
        batch_metrics = metric_collection.forward(pred_change, labels)  # compute metric
        if i == sample_batch:
        # log loss and metric
            logger = open(logFileLoc, 'a')
            logger.write(f'''
            {mode} loss': {cd_loss}
            {mode} iou': {batch_metrics['iou']}
            {mode} precision': {batch_metrics['precision']}
            {mode} recall': {batch_metrics['recall']}
            {mode} f1score': {batch_metrics['f1score']}
            learning rate: {optimizer.param_groups[0]['lr']}
            {mode} loss_change': {loss[0]},
            {mode} loss_seg1': {loss[1]},
            {mode} loss_seg2': {loss[2]},
            {mode} loss_seg_all': {loss[3]},
             'step': {total_step},
             'epoch': {epoch}
            ''')
            logger.flush()
            logger.close()
        del batch_img1, batch_img2, labels,label_t1_seg, label_t2_seg,label_all_seg,weight_t1,weight_t2,weight_all
    epoch_metrics = metric_collection.compute()  # compute epoch metric
    epoch_loss /= n_iter
    logger = open(logFileLoc, 'a')
    for k in epoch_metrics.keys():
        logger.write(f'''epoch_{mode}_{str(k)}': {epoch_metrics[k]},
                       epoch: {epoch}
        ''')  # log epoch metric
        logger.flush()
    metric_collection.reset()
    logger.write(f'''epoch_{mode}_loss': {epoch_loss},
                   'epoch': {epoch}''')  # log epoch loss
    logger.flush()
    logger.close()
    if mode == 'val':
        # print('237')
        if epoch_metrics['f1score'] > best_metrics['best_f1score']:
            non_improved_epoch = 0
            best_metrics['best_f1score'] = epoch_metrics['f1score']
            if ph.save_best_model:
                save_model(net, best_f1score_model_path, epoch, 'f1score')
        elif epoch_loss < best_metrics['lowest loss']:
            best_metrics['lowest loss'] = epoch_loss
            if ph.save_best_model:
                save_model(net, best_loss_model_path, epoch, 'loss')
        else:
            non_improved_epoch += 1
            if non_improved_epoch == ph.patience:
                lr *= ph.factor
                for g in optimizer.param_groups:
                    g['lr'] = lr
                non_improved_epoch = 0
        if (epoch + 1) % ph.save_interval == 0 and ph.save_checkpoint:
            save_model(net, checkpoint_path, epoch, 'checkpoint', optimizer=optimizer)
        # print('258')
    if mode == 'train':
        return  net, optimizer, grad_scaler, total_step, lr
    elif mode == 'val':
        return  net, optimizer, total_step, lr, best_metrics, non_improved_epoch
    else:
        raise NameError('mode should be train or val')
