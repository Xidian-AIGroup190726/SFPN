import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from utils.path_hyperparameter import ph
import cv2
from torchvision import transforms as T
from pathlib import Path
from kornia.filters import SpatialGradient
from torch import Tensor
from mmengine.model import BaseModule
from einops import rearrange
import typing as t
class EDA(nn.Module):
    def __init__(self,in_channel):
        super(EDA, self).__init__()
        self.k = kernel_size(in_channel)
        self.spatial = SpatialGradient('diff')
        self.max_pool = nn.MaxPool2d(self.k, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        y = self.max_pool(u)
        output=u
        return output


class BS(nn.Module):
    def __init__(self,in_channel):
        super(BS,self).__init__()
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel*3 , out_channels=in_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def cosine_similarity(self, x1, x2):
        """Calculate cosine similarity between two tensors."""
        dot_product = (x1 * x2).sum(dim=-1, keepdim=True)
        norm1 = torch.norm(x1, dim=-1, keepdim=True)
        norm2 = torch.norm(x2, dim=-1, keepdim=True)
        # 防止除零错误
        norm1 = torch.max(norm1, torch.tensor(1e-8).to(x1.device))
        norm2 = torch.max(norm2, torch.tensor(1e-8).to(x2.device))
        return dot_product / (norm1 * norm2)

    def l2_normalize(self, tensor, dim=1):
        """Perform L2 normalization on a tensor along the specified dimension."""
        norm = tensor.norm(p=2, dim=dim, keepdim=True)
        norm = torch.max(norm, torch.tensor(1e-8).to(tensor.device))
        normalized_tensor = tensor.div(norm.expand_as(tensor))
        return normalized_tensor

    def forward(self, x1, x2,log=None,module_name=None, img_name=None):

        x1_normalized = self.l2_normalize(x1, dim=1 )
        x2_normalized = self.l2_normalize(x2, dim=1 )
        x_sub = torch.abs(x1_normalized - x2_normalized)

        x_sub = self.l2_normalize(x_sub, dim=1 )
        x1_permuted = x1_normalized.permute(0, 2, 3, 1)  # shape [N, H, W, C]
        x2_permuted = x2_normalized.permute(0, 2, 3, 1)  # shape [N, H, W, C]
        x_sub_permuted = x_sub.permute(0, 2, 3, 1)  # shape [N, H, W, 1]

        cosine_similarity1 = self.cosine_similarity(x1_permuted, x_sub_permuted)  # shape [N, H, W, 1]
        cosine_similarity2 = self.cosine_similarity(x2_permuted, x_sub_permuted)
        cosine_similarity1 = cosine_similarity1.permute(0, 3, 1, 2)  # shape [N, 1, H, W]
        cosine_similarity2 = cosine_similarity2.permute(0, 3, 1, 2)  # shape [N, 1, H, W]
        cosine_similarity1=cosine_similarity1+2
        cosine_similarity2=cosine_similarity2+2
        s=torch.abs(cosine_similarity1 - cosine_similarity2)+1
        w1=s*cosine_similarity1/(cosine_similarity1+cosine_similarity2)
        w2=s*cosine_similarity2/(cosine_similarity1 + cosine_similarity2)
        w=(ph.arf-s/(cosine_similarity1+cosine_similarity2))*s
        input=torch.cat([w*x_sub,w1*x1,w2*x2], dim=1)
        output=self.fuse(input)
        if log:
            log_list = [x1, x2, cosine_similarity1, cosine_similarity2, output]
            feature_name_list = ['x1', 'x2', 'cosine_similarity1', 'cosine_similarity2', 'output']
            log_feature(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return output



class Conv_BN_ReLU(nn.Module):
    """ Basic convolution."""

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True),
                                          )

    def forward(self, x):
        output = self.conv_bn_relu(x)

        return output


class CGSU(nn.Module):
    """Basic convolution module."""

    def __init__(self, in_channel):
        super().__init__()

        mid_channel = in_channel // 2

        self.conv1 = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )

    def forward(self, x):
        x1, x2 = channel_split(x)
        x1 = self.conv1(x1)
        output = torch.cat([x1, x2], dim=1)

        return output


class CGSU_DOWN(nn.Module):
    """Basic convolution module with stride=2."""

    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,
                                             stride=2, bias=False),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )
        self.conv_res = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # remember the tensor should be contiguous
        output1 = self.conv1(x)

        # respath
        output2 = self.conv_res(x)

        output = torch.cat([output1, output2], dim=1)

        return output


class Changer_channel_exchange(nn.Module):
    """Exchange channels of two feature uniformly-spaced with 1:1 ratio."""

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask1 = exchange_mask.cuda().int().expand((N, C)).unsqueeze(-1).unsqueeze(-1)  # b,c,1,1
        exchange_mask2 = 1 - exchange_mask1
        out_x1 = exchange_mask1 * x1 + exchange_mask2 * x2
        out_x2 = exchange_mask1 * x2 + exchange_mask2 * x1

        return out_x1, out_x2

class DPFA(nn.Module):
    """Fuse two feature into one feature."""

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=None, module_name=None,
                img_name=None):
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        if log:
            log_list = [t1, t2, t1_spatial_attention, t2_spatial_attention, fuse]
            feature_name_list = ['t1', 't2', 't1_spatial_attention', 't2_spatial_attention', 'fuse']
            log_feature(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return fuse


class atten_model(BaseModule):

    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(atten_model, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4
        #卷积3、5、7、9
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        self.fuse_conv = nn.Sequential(
            nn.Conv1d(2 * dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = x.size()
        x_avg_h = x.mean(dim=3)  # B, C, H
        x_max_h, _ = x.max(dim=3)  # B, C, H
        x_h = torch.cat([x_avg_h, x_max_h], dim=1)
        x_h = self.fuse_conv(x_h)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_avg_w = x.mean(dim=2)  # B, C, W
        x_max_w, _ = x.max(dim=2)  # B, C, W
        x_w = torch.cat([x_avg_w, x_max_w], dim=1)  # B, 2C, W
        x_w=self.fuse_conv(x_w)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn
        y = self.down_func(x)
        y = self.conv_d(y)

        _, _, h_, w_ = y.size()
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x

class conv_atten(nn.Module):

    def __init__(self, in_channel):
        super().__init__()

        dim = in_channel

        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv3_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=dim * 4, out_channels=dim,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(dim),
                                  )
        self.sigmoid = nn.Sigmoid()

        self.k = kernel_size(dim)
        self.channel_conv = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.atten_model = atten_model(dim=in_channel, head_num=2)
    def forward(self, x):
        u = x.clone()
        #3*3
        attn = self.conv0(x)
        #多尺度卷积
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)
        attn = self.conv3(torch.cat([attn_0, attn_1,attn_2,attn_3], dim=1))
        attn=self.sigmoid(attn)
        output1=attn * u
        #atten_model
        output = self.atten_model(output1)

        return output

class Encoder_Block(nn.Module):
    """ Basic block in encoder"""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel * 2, 'the out_channel is not in_channel*2 in encoder block'
        self.conv1 = nn.Sequential(
            CGSU_DOWN(in_channel=in_channel),
            CGSU(in_channel=out_channel),
            CGSU(in_channel=out_channel)
        )
        self.conv3 = Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)
        self.conv_atten = conv_atten(in_channel=out_channel)

    def forward(self, x, log=False, module_name=None, img_name=None):
        x = self.conv1(x)
        x = self.conv3(x)
        x_res = x.clone()
        if log:
            output = self.conv_atten(x)
        else:
            output = self.conv_atten(x)
        output = x_res + output

        return output


class Decoder_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output


def kernel_size(in_channel):
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


def channel_split(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


def log_feature(log_list, module_name, feature_name_list, img_name, module_output=True):
    for k, log in enumerate(log_list):
        log = log.clone().detach()
        b, c, h, w = log.size()
        if module_output:
            log = torch.mean(log, dim=1, keepdim=True)
            log = F.interpolate(
                log * 255, scale_factor=ph.patch_size // h,
                mode='nearest').reshape(b, ph.patch_size, ph.patch_size, 1) \
                .cpu().numpy().astype(np.uint8)
            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_equalize_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '_equalize/'
            Path(log_equalize_dir).mkdir(parents=True, exist_ok=True)

            for i in range(b):
                log_i = cv2.applyColorMap(log[i], cv2.COLORMAP_JET)
                cv2.imwrite(log_dir + img_name[i] + '.png', log_i)

                log_i_equalize = cv2.equalizeHist(log[i])
                log_i_equalize = cv2.applyColorMap(log_i_equalize, cv2.COLORMAP_JET)
                cv2.imwrite(log_equalize_dir + img_name[i] + '.png', log_i_equalize)
        else:
            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log = torch.round(log)
            log = F.interpolate(log, scale_factor=ph.patch_size // h,
                                mode='nearest').cpu()
            to_pil_img = T.ToPILImage(mode=None)
            for i in range(b):
                log_i = to_pil_img(log[i])
                log_i.save(log_dir + img_name[i] + '.png')
