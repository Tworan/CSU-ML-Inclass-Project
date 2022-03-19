import cupy as cp 
import numpy as np
import os
import sys 
sys.path.append('/yourpath/NN/Photon')
from utils import *


# 在里面传播的数据只能是cupy数组 即保存在GPU中
'''
注意：
    如果输入图片的话，一定要确保输入的axis的顺序是 (Samples, H, W, Channels) 
    如果输入是多维变量，则确保 (Samples, features)
'''
# 激活函数
class ReLU:
    def __init__(self):
        self.dead_nodes = None 

    def forward(self, x):
        self.dead_nodes = (x <= 0)
        out = x.copy()
        out[self.dead_nodes] = 0.
        return out

    def backward(self, dout):
        dout[self.dead_nodes] = 0.
        return dout 

class Sigmoid:
    def __init__(self):
        self.out = None 
        self.sigmoid = lambda x: 1. / (1 + cp.exp(-x)) 

    def forward(self, x):
        self.out = self.sigmoid(x)
        return self.out 

    def backward(self, dout):
        # dx = y(1-y)
        return dout * (1. - self.out) * self.out 
    
class Affine:
    def __init__(self, W, b):
        self.W = W  
        self.b = b 
        
        self.x = None 
        self.x_shape = None 
        # 记录反向传播的信息
        self.dW = None 
        self.db = None 

    def forward(self, x):
        self.x_shape = x.shape 
        # Flatten处理 
        # 因为处理的是一个batch 有很多个样本
        x = x.reshape(x.shape[0], -1)
        self.x = x 
        out = cp.dot(self.x, self.W) + self.b 
        return out  
    
    def backward(self, dout):
        # 链式法则
        dx = cp.dot(dout, self.W.T)
        # 保留梯度
        self.dW = cp.dot(self.x.T, dout) 
        self.db = cp.sum(dout, axis=0) 
        # 还原数据原始形状
        dx = dx.reshape(self.x_shape)
        return dx

# 损失函数 这里就放一个好了
class SoftmaxLossLayer:
    ''' 
    注意：这里的输入要是one-hot编码的形式
    '''
    def __init__(self):
        self.loss = None
        # y_true and y_pred
        self.y_true = None 
        self.y_pred = None  
        self.batch_size = None
    
    def adjusted_softmax(self, x):
        # 防止溢出
        x = x.T
        x = x - cp.max(x, axis=0)
        y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
        return y.T 
    
    def crossentropy(self, y_pred, y_true):
        # 注意 输入的y_true和y_pred 要是one-hot编码的形式
        return - cp.sum(y_true * cp.log(y_pred + 1e-7)) / y_true.shape[0]

    def forward(self, x, y_true):
        self.y_true = y_true
        self.y_pred = self.adjusted_softmax(x) 
        self.batch_size = x.shape[0]
        self.loss = self.crossentropy(self.y_pred, self.y_true)
        return self.loss
    
    def backward(self):
        '''
        dout: 1
        '''
        dx = (self.y_pred - self.y_true) / self.batch_size
        return dx

# Dropout 防止过拟合
class Dropout:
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.mask = None 

    def forward(self, x, train=True):
        if train:
            self.mask = cp.random.rand(*x.shape) > self.ratio 
            return x * self.mask 
        
        else: 
            # 向前传播期望
            return x * (1. - self.ratio) 
    
    def backward(self, dout): 
        return dout * self.mask 
    
# BatchNormalization 
class BatchNormalization:
    '''
    注意: 这里要分两种情况，一种输入的时候是四维张量 即一张图片， 另一种是输入的是二维的
    '''
    def __init__(self, gamma, beta, momentum=0.1): 
        self.gamma = gamma 
        self.beta = beta
        self.momentum = momentum 
        self.input_type = None # 1-> 图片 2-> 多变量
        self.initialization = True
        self.shape = None

        # 这是训练时候的渐进均值 
        self.running_var = None
        self.running_mean = None

        # backward
        self.batch_size = None 
        self.normalized_x = None
        self.std = None 
        self.xc = None
        self.dg = None
        self.dbeta = None

    def forward(self, x, train=True):
        '''
        只有在训练时才会更新渐进的值
        '''
        # 首先先确定输入的type
        # 初始化

        self.shape = x.shape
        if len(x.shape) == 2:
            self.input_type = 1
            self.batch_size = self.shape[0]

        elif len(x.shape) == 4:
            self.input_type = 2
            self.batch_size = self.shape[0]
            x = x.reshape(self.batch_size, -1)
        if self.initialization:
            self.running_mean = cp.zeros(x.shape[1])
            self.running_std = cp.zeros(x.shape[1])
            self.initialization = True
    
        if train:
            mean = x.mean(axis=0) 
            std = x.std(axis=0)
            self.xc = x - mean
            self.normalized_x = (x - mean) / (std + 1e-7)
            self.std = std 
            # 渐进均值和方差估计
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean  
            self.running_std = self.momentum * self.running_std + (1 - self.momentum) * std     

        else: 
            self.normalized_x = (x - self.running_mean) / (self.running_std + 1e-7)
        
        out = self.gamma * self.normalized_x + self.beta 
        return out.reshape(self.shape) 

    def backward(self, dout):
        if self.input_type == 2:
            dout = dout.reshape(self.batch_size, -1)
        # 这一大堆老复杂了 看晕了
        dbeta = dout.sum(axis=0)
        dgamma = cp.sum(self.normalized_x * dout, axis=0) 
        d_normalized_x = self.gamma * dout 
        dxc = d_normalized_x / (self.std + 1e-7)
        dstd = -cp.sum(d_normalized_x * self.xc / (self.std + 1e-7) / (self.std + 1e-7), axis=0) 
        dvar = 0.5 * dstd / (self.std + 1e-7)  
        dxc += (2. / self.batch_size) * self.xc * dvar
        dmean = cp.sum(dxc, axis=0) 
        dx = dxc - dmean / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx 

# 二维卷积 这里的W就相当于是卷积核
class Conv2D: 
    def __init__(self, W, b, strides=(1, 1), padding='None'):
        '''
        W 是卷积核 要求 (Samples, H, W, Channels)
        b 是偏置
        padding = {'None', 'same'}
        卷积核kernel_size的输入 必须要是奇数！
        '''

        self.W = W  
        self.b = b  
        self.strides = strides
        self.padding = padding
        if self.padding == 'same':
            self.pads = 1 
        else: 
            self.pads = 0

        self.out_h = None 
        self.out_w = None

        self.x = None 
        self.col = None  
        self.col_W = None  

        self.dW = None 
        self.db = None
    
    def forward(self, x):
        Filter_num, Filter_H, Filter_W, Filter_channel = self.W.shape  
        Samples_num, H, W, channel = x.shape 
        self.x = x
        # 计算输出的尺寸
        if not self.out_h:
            if self.pads:
                # pads 向下取整
                self.out_h = H // self.strides[0]
                self.out_w = W // self.strides[1]
            else:
                self.out_h = (H - 2 * Filter_H // 2) // self.strides[0] 
                self.out_w = (W - 2 * Filter_W // 2) // self.strides[1]

        col = img2col(x, Filter_H, Filter_W, self.strides, self.out_h, self.out_w)
        col_W = self.W.reshape(Filter_num, -1).T 
        out = cp.dot(col, col_W) + self.b 
        out = out.reshape(Samples_num, self.out_h, self.out_w, -1)
        self.col = col 
        self.col_W = col_W
        return out

    def backward(self, dout):
        Filter_num, Filter_H, Filter_W, Filter_channel = self.W.shape  
        dout = dout.reshape(-1, Filter_num)

        self.db = cp.sum(dout, axis=0) 
        self.dW = cp.dot(self.col.T, dout).transpose(1, 0).reshape(Filter_num, Filter_H, Filter_W, Filter_channel)

        dcol = cp.dot(dout, self.col_W.T) 
        dx = col2img(dcol, self.x.shape, Filter_H, Filter_W, self.strides, self.out_h, self.out_w)        
        return dx 

# MaxPooling2D比较好写 所以写这玩意
# MaxPooling2D 可以放在CPU上运行
class MaxPooling2D: 
    def __init__(self, kernel_size=(2, 2), strides=(2, 2), padding='None'): 
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        if self.padding == 'None':
            self.pads = 0
        elif self.padding == 'same':
            self.pads = 1
        else:
            raise ValueError
        
        self.x = None  
        self.max_args = None 

        self.out_h = None 
        self.out_w = None 

    def forward(self, x): 
        Samples_num, H, W, channel = x.shape 
        # 计算输出的尺寸
        if not self.out_h:
            if self.pads:
                # pads 向下取整
                self.out_h = H // self.strides[0]
                self.out_w = W // self.strides[1]
            else:
                self.out_h = (H - 2 * (self.kernel_size[0] // 2)) // self.strides[0] 
                self.out_w = (W - 2 * (self.kernel_size[1] // 2)) // self.strides[1]

        col = img2col(x, self.kernel_size[0], self.kernel_size[1], self.strides, self.out_h, self.out_w)
        col = col.reshape(-1, self.kernel_size[0] * self.kernel_size[1]) 
        
        max_args = cp.argmax(col, axis=1) 
        out = cp.max(col, axis=1) 
        out = out.reshape(Samples_num, self.out_h, self.out_w, channel)        
        
        self.x = x.copy()
        self.max_args = max_args 
        return out 

    def backward(self, dout):
        kernel_area = self.kernel_size[0] * self.kernel_size[1]

        dmaxargs = cp.zeros((dout.size, kernel_area))
        dmaxargs[np.arange(self.max_args.size), self.max_args.flatten()] = dout.flatten() 
        dmaxargs = dmaxargs.reshape(dout.shape + (kernel_area, ))
        # 拍平处理
        dcol = dmaxargs.reshape(dmaxargs.shape[0] * dmaxargs.shape[1] * dmaxargs.shape[2], -1)
        dx = col2img(dcol, self.x.shape, self.kernel_size[0], self.kernel_size[1], self.strides, self.out_h, self.out_w)
        
        return dx 

class Add:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out.copy()
    
    def backward(self, dout):
        return dout.copy(), dout.copy()

class Diversion:
    def __init__(self):
        pass
    
    def forward(self, x):
        return x.copy(), x.copy()
    
    def backward(self, dout1, dout2):
        dout = dout1 + dout2 
        return dout