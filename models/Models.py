import torch.nn as nn
from torch.nn import init
from models.dpcd_parts import (Conv_BN_ReLU, CGSU, Encoder_Block, DPFA, Decoder_Block,EDA,BS,
                               Changer_channel_exchange, log_feature)
import torch
from utils.path_hyperparameter import ph
class DPCD(nn.Module):
    def __init__(self):
        super().__init__()
        channel_list = [8, 16, 32, 64, 128]
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1),
                                       CGSU(in_channel=channel_list[0]),
                                       CGSU(in_channel=channel_list[0]))
        self.en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3])
        self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4])

        self.channel_exchange4 = Changer_channel_exchange()
        # decoder
        self.de_block1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.de_block2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.de_block3 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])
        #EDA
        self.eda1 = EDA(in_channel=channel_list[3])
        self.eda2 = EDA(in_channel=channel_list[2])
        self.eda3 = EDA(in_channel=channel_list[1])
        #seg_all
        self.seg_all= nn.Conv2d(in_channels=channel_list[1]*2, out_channels=channel_list[1], kernel_size=3, stride=1, padding=1)
        # dpfa
        self.dpfa1 = DPFA(in_channel=channel_list[4])
        self.dpfa2 = DPFA(in_channel=channel_list[3])
        self.dpfa3 = DPFA(in_channel=channel_list[2])
        self.dpfa4 = DPFA(in_channel=channel_list[1])

        self.change_block4 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.change_block3 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.change_block2 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.seg_out1 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)
        self.seg_out2 = nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)
        self.seg_out_all= nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1)

        self.bs1 = BS(in_channel=channel_list[4])
        self.bs2 = BS(in_channel=channel_list[3])
        self.bs3 = BS(in_channel=channel_list[2])
        self.bs4 = BS(in_channel=channel_list[1])

        self.upsample_x2_1 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.upsample_x2_2 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.upsample_x2_3 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.upsample_x2_4 = nn.Sequential(
            nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.conv_out_change = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, t1, t2, log=False, img_name=None):
        # encoder
        t1_1 = self.en_block1(t1)#3->l0
        t2_1 = self.en_block1(t2)

        if log:
            t1_2 = self.en_block2(t1_1, log=log, module_name='t1_1_en_block2', img_name=img_name)#l0->l1
            t2_2 = self.en_block2(t2_1, log=log, module_name='t2_1_en_block2', img_name=img_name)

            t1_3 = self.en_block3(t1_2, log=log, module_name='t1_2_en_block3', img_name=img_name)#l1->l2
            t2_3 = self.en_block3(t2_2, log=log, module_name='t2_2en_block3', img_name=img_name)

            t1_4 = self.en_block4(t1_3, log=log, module_name='t1_3_en_block4', img_name=img_name)#l2->l3
            t2_4 = self.en_block4(t2_3, log=log, module_name='t2_3_en_block4', img_name=img_name)
            t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)

            t1_5 = self.en_block5(t1_4, log=log, module_name='t1_4_en_block5', img_name=img_name)#l3->l4
            t2_5 = self.en_block5(t2_4, log=log, module_name='t2_4_en_block5', img_name=img_name)
        else:
            t1_2 = self.en_block2(t1_1)
            t2_2 = self.en_block2(t2_1)

            t1_3 = self.en_block3(t1_2)
            t2_3 = self.en_block3(t2_2)

            t1_4 = self.en_block4(t1_3)
            t2_4 = self.en_block4(t2_3)
            t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)

            t1_5 = self.en_block5(t1_4)
            t2_5 = self.en_block5(t2_4)

        de1_5 = t1_5
        de2_5 = t2_5

        de1_4 = self.de_block1(de1_5, t1_4)  # l4->l3
        de2_4 = self.de_block1(de2_5, t2_4)

        de1_3 = self.de_block2(de1_4, t1_3)  # l3->l2
        de2_3 = self.de_block2(de2_4, t2_3)

        de1_2 = self.de_block3(de1_3, t1_2)  # l2->l1
        de2_2 = self.de_block3(de2_3, t2_2)

        de1 = self.upsample_x2_1(de1_2) #l1->8
        de2 = self.upsample_x2_2(de2_2)
        seg_out1 = self.seg_out1(de1) #8->1
        seg_out2 = self.seg_out2(de2)


        seg_all=self.seg_all(torch.cat([de1_2,de2_2],dim=1)) #list[1]*2->list[1]
        seg_all=self.upsample_x2_3(seg_all)
        seg_all_out=self.seg_out_all(seg_all)
        if log:
            change_5 = self.bs1(de1_5, de2_5, log=log, module_name='de1_5_de2_5_bs1',
                                  img_name=img_name)

            change_4 = self.change_block4(change_5, self.bs2(de1_4, de2_4, log=log, module_name='de1_4_de2_4_bs2',
                                                               img_name=img_name))

            change_3 = self.change_block3(change_4, self.bs3(de1_3, de2_3, log=log, module_name='de1_3_de2_3_bs3',
                                                               img_name=img_name))

            change_2 = self.change_block2(change_3, self.bs4(de1_2, de2_2, log=log, module_name='de1_2_de2_2_bs4',
                                                               img_name=img_name))
        else:
            change_5 = self.bs1(de1_5, de2_5, log=log, module_name='de1_5_de2_5_bs1',
                                  img_name=img_name)

            change_4 = self.change_block4(change_5, self.bs2(de1_4, de2_4))

            change_3 = self.change_block3(change_4, self.bs3(de1_3, de2_3))

            change_2 = self.change_block2(change_3, self.bs4(de1_2, de2_2))

        change = self.upsample_x2_4(change_2)
        # change_out = self.conv_out_change(change)
        change_out = torch.sigmoid(self.conv_out_change(change))

        if log:
            log_feature(log_list=[change_out, seg_out1, seg_out2,seg_all_out], module_name='model',
                        feature_name_list=['change_out', 'seg_out1', 'seg_out2','seg_all_out'],
                        img_name=img_name, module_output=False)

        return change_out, seg_out1, seg_out2 ,seg_all_out
