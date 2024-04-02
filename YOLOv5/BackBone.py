import torch
import torch.nn as nn
import time
#定义一个平滑版本的relu函数
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x*torch.sigmoid(x)

def autopad(k,p=None):
    #使得3x3的卷积一直都会让输出的尺寸不变 比如k==3 输出的1作为padding
    #当步长为2时，同样适用 例如640的尺寸 k=3，s=2 计算padding=1 （640+2-3）//2+1=320
    if p is None:
        p=k//2 if isinstance(k,int) else [x//2 for x in k]
    return p

class Focus(nn.Module):
    #img输入的第一个网络层就是focus层
    #输出通道增加
    def __init__(self,c1,c2,k=1,s=1,p=None,g=1,act=True):
        super(Focus,self).__init__()
        self.conv=Conv(c1*4,c2,k,s,p,g,act)

    def forward(self,x):
       #320,320,12 ->320,320,64
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ], 1
            )  #将分开的四个原图信息再通道维度进行堆叠
        )

class Conv(nn.Module):  #yolov5中的卷积块都是1卷积+归一化+激活
    def __init__(self,c1,c2,k=1,s=1,p=None,g=1,act=True):
        super(Conv,self).__init__()
        #该网络层是一个1卷积层，同时归一化和silu激活
        self.conv   =nn.Conv2d(c1,c2,k,s,autopad(k,p),groups=g,bias=False)
        self.bn     =nn.BatchNorm2d(c2,eps=0.001,momentum=0.03)
        self.act    =SiLU()if act is True else (act if isinstance((act,nn.Module)) else nn.Identity())

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

    def fushforward(self,x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):  #类似残差的实现
    def __init__(self,c1,c2,shortcut=True,g=1,e=0.5):
        super(Bottleneck,self).__init__()
        c_=int(c2*e)
        self.cv1=Conv(c1,c_,1,1)  # 1 卷积
        self.cv2=Conv(c_,c2,3,1,g=g)   # 输出尺寸依然是不变的
        self.add= shortcut and c1==c2    #是否“残差”连接
    def forward(self,x):
        return x+self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):  #连续“残差”
    def __init__(self,c1,c2,n=1,shortcut=True,g=1,e=0.5):
        super(C3,self).__init__()
        c_=int(c2*e)
        #若e=1了，那么1卷积就相当于做了一次线性变化
        self.cv1=Conv(c1,c_,1,1)
        self.cv2=Conv(c1,c_,1,1)
        self.cv3=Conv(2*c_,c2,1)
        self.m=nn.Sequential(*[Bottleneck(c_,c_,shortcut,g,e=1.0)for _ in range(n)])

    def forward(self,x):   #类似残差连接 x+g（x） x仅作一个卷积 g（x）是卷积后再残差了n次
        return self.cv3(torch.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            )
            , dim=1))

class SPP(nn.Module):#不同池化大小的结果（特征概括的范围）平叠
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP,self).__init__()
        c_=c1//2
        self.cv1=Conv(c1,c_,1,1)
        self.cv2=Conv(c_*(len(k)+1),c2,1,1)
        self.m=nn.ModuleList([nn.MaxPool2d(kernel_size=x,stride=1,padding=x//2) for x in k])
        # maxpool的输出尺寸是不会变的
    def forward(self,x):
        x=self.cv1(x)
        return self.cv2(torch.cat([x]+[m(x) for m in self.m],1))  #cv2的输入是通道数是x+三个x=4个x，img的尺寸不变

class CSPDarknet(nn.Module):
    def __init__(self,base_channels,base_depth=1):
        super().__init__()
        # base_channels = 64
        # 640,640,3 ->320,320,12->320,320,64
        self.stem=Focus(3,base_channels,k=3)
        self.dark2=nn.Sequential(
            #from 64,320,320 to 128,160,160
            Conv(base_channels,base_channels*2,3,2),

            #from 128,160,160 to 128,160,160 不变，但是经过了好多个残差网络提取到了非常多的特征
            C3(base_channels*2,base_channels*2,base_depth),
        )

        self.dark3=nn.Sequential(
            Conv(base_channels*2,base_channels*4,3,2),
            C3(base_channels*4,base_channels*4,base_depth*3),
            #先进行一次卷积 降了一半的大小 然后进行多次的“残差”连接，
        )
        #这里输出的尺寸应该是 256，80，80

        self.dark4=nn.Sequential(
            Conv(base_channels*4,base_channels*8,3,2),
            C3(base_channels*8,base_channels*8,base_depth*3),
        )
        #这里输出的结果应该是 521，40，40

        self.dark5=nn.Sequential(
            Conv(base_channels*8,base_channels*16,3,2),
            SPP(base_channels * 16, base_channels * 16),
            C3(base_channels*16,base_channels*16,base_depth,shortcut=False),
        )

    def forward(self,x):
        #Focus
        x=self.stem(x)
        x=self.dark2(x)

        x=self.dark3(x)
        feat1=x   #输出第一个特征层 256,80,80

        x=self.dark4(x)
        feat2=x  #输出第二个特征层 521，40，40

        x=self.dark5(x)
        feat3=x  #输出第三个特征层 1042，20，20

        return feat1,feat2,feat3

# x=torch.randn(16,3,640,640)
# start=time.time()
# net=CSPDarknet(3,64)
# y1,y2,y3=net(x)
# print(y1.shape)
# print(y2.shape)
# print(y3.shape)
# end=time.time()
# print(end-start)