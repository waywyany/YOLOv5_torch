import torch
import torch.nn as nn
from BackBone import Conv,C3,CSPDarknet
import time
import numpy as np
class YoloBody(nn.Module):
    def __init__(self,anchors_mask,num_classes,input_shape=[640,640]):
        super(YoloBody,self).__init__()
        base_channels = 64
        base_depth=3
        self.backbone=CSPDarknet(base_channels,base_depth)
        self.upsample =nn.Upsample(scale_factor=2,mode="nearest")

        self.conv_for_feat3 =Conv(base_channels*16,base_channels*8,1,1)
        self.conv3_for_upsample1 = C3(base_channels*16,base_channels*8,base_depth,shortcut=False)
        #这里注意，shortcut无了，意味着不再“残差”连接了

        self.conv_for_feat2 =Conv(base_channels*8,base_channels*4,1,1)
        self.conv3_for_upsample2 = C3(base_channels*8,base_channels*4,base_depth,shortcut=False)

        self.down_sample1 = Conv(base_channels*4,base_channels*4,3,2)
        self.conv3_for_down_sample1 = C3(base_channels*8,base_channels*8,base_depth,shortcut=False)

        self.down_sample2 = Conv(base_channels*8,base_channels*8,3,2)
        self.conv3_for_down_sample2 = C3(base_channels*16,base_channels*16,base_depth,shortcut=False)

        self.yolo_head_p3 = nn.Conv2d(base_channels*4,len(anchors_mask)*(5+num_classes),1)
        self.yolo_head_p4 = nn.Conv2d(base_channels*8,len(anchors_mask)*(5+num_classes),1)
        self.yolo_head_p5 = nn.Conv2d(base_channels*16,len(anchors_mask)*(5+num_classes),1)
        # self.yolo_head_p3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # self.yolo_head_p4 = nn.Conv2d(base_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        # self.yolo_head_p5 = nn.Conv2d(base_channels * 16, len(anchors_mask[2]) * (5 + num_classes), 1)

    def forward(self,x):
        feat1,feat2,feat3=self.backbone(x)
        #256,512,1024
        p5= self.conv_for_feat3(feat3)  #1024 ->512 尺寸不变
        p5_upsample=self.upsample(p5)   #尺寸乘2  20,20 -> 40,40

        p4 =torch.cat([p5_upsample,feat2],1)   #40，40，（512+512）
        p4 =self.conv3_for_upsample1(p4)     #1024->512
        p4 =self.conv_for_feat2(p4)      #继续降通道 512->256
        p4_upsample=self.upsample(p4)    #尺寸乘2  40,40,256 ->80,80,256

        p3 =torch.cat([p4_upsample,feat1],1)  # 平叠 80,80,(256+256)
        p3 =self.conv3_for_upsample2(p3)  # 512->256
        p3_downsample = self.down_sample1(p3)  #降采样 80,80,256-> 40,40,256

        p4=torch.cat([p3_downsample,p4],1)    #40,40,(256+256)
        p4=self.conv3_for_down_sample1(p4)   #全连接层，通道数没有变

        p4_downsample =self.down_sample2(p4)  #降采样 20,20,512
        p5=torch.cat([p4_downsample,p5],1)   #20,20,(512+512)
        p5=self.conv3_for_down_sample2(p5)  #全连接层 啥都没变

        out2=self.yolo_head_p3(p3)

        out1=self.yolo_head_p4(p4)

        out0=self.yolo_head_p5(p5)

        return out0,out1,out2


# x=torch.randn(1,3,640,640)
# ab=[1,2,3]  #假设有先验框的个数是3个
# start=time.time()
# net=YoloBody(ab,80)
# y0,y1,y2=net(x)
# print(y0.shape)
# print(y1.shape)
# print(y2.shape)