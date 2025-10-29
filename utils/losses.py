import torch
import torch.nn as nn
from utils.path_hyperparameter import ph


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)


class dice_focal_loss(nn.Module):

    def __init__(self):
        super(dice_focal_loss, self).__init__()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels):
        diceloss = self.binnary_dice(scores.clone(), labels)
        if torch.isnan(diceloss):
            print("Dice Loss contains NaN.")
            print("Scores:")
            print(scores)
            print("Labels:")
            print(labels)
            raise ValueError("Dice Loss is NaN")
        foclaloss = self.focal_loss(scores.clone(), labels)
        if torch.isnan(foclaloss):
            print("Focal Loss contains NaN.")
            print("Scores:")
            print(scores)
            print("Labels:")
            print(labels)
            raise ValueError("Focal Loss is NaN")
        # return [diceloss, foclaloss]
        return diceloss+foclaloss

def dynamic_weighting(losses, initial_weights=[1, ph.beta, ph.beta, ph.beta]):
    assert all(isinstance(loss, torch.Tensor) for loss in losses), "All losses must be tensors."
    avg_losses = [torch.mean(loss) for loss in losses]
    variances = [torch.var(loss, unbiased=False) for loss in losses]
    epsilon = 1
    inv_variances = [1.0 / (var + epsilon + avg) for var, avg in zip(variances, avg_losses)]
    adjusted_weights = [w * iv for w, iv in zip(initial_weights, inv_variances)]
    adjusted_weights_sum = sum(adjusted_weights)
    adjusted_weights = [w / adjusted_weights_sum for w in adjusted_weights]
    combined_loss = [w * torch.mean(l) for w, l in zip(adjusted_weights, losses)]
    if any(torch.isnan(cl).any() for cl in combined_loss):
        print("Combined loss contains NaN.")
        print(losses)
        print(combined_loss)
        raise Exception("NaN detected")
    return combined_loss


def loss_calc(pred_flux, gt_flux, weight_matrix):
    device_id = pred_flux.device
    weight_matrix = weight_matrix.cuda(device_id)
    gt_flux = gt_flux.cuda(device_id)

    gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1, keepdim=True) + 1e-9)
    weight_matrix_unsqueeze = weight_matrix.unsqueeze(1)
    # norm loss
    norm_loss = weight_matrix_unsqueeze * (pred_flux - gt_flux) ** 2
    norm_loss = norm_loss.sum()

    # angle loss
    pred_flux = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1, keepdim=True) + 1e-9)
    angle_loss = (torch.acos(torch.sum(pred_flux * gt_flux, dim=1)))**2
    angle_loss = angle_loss.sum()+norm_loss
    return angle_loss


def FCCDN_loss_without_seg(score0,score1,score2,score3,label0,label1,label2,label3, weight_matrix_1, weight_matrix_2, weight_matrix_3):
    score0 = score0.squeeze(1) if len(score0.shape) > 3 else score0
    label0 = label0.squeeze(1) if len(label0.shape) > 3 else label0
    """ for binary change detection task"""
    criterion_change = dice_focal_loss()
    loss_change = criterion_change(score0, label0)
    loss_seg1 = loss_calc(score1, label1, weight_matrix_1)
    loss_seg2 = loss_calc(score2, label2, weight_matrix_2)
    loss_seg_all = loss_calc(score3, label3, weight_matrix_3)
    losses = [loss_change, loss_seg1, loss_seg2, loss_seg_all]

    # return losses
    combined_loss = dynamic_weighting(losses)
    return combined_loss
