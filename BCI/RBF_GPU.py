import os, sys 
sys.path.append('/home/oneran/机器学习课设/cifar-10/Photon')
import numpy as cp 
import cupy as cp 
import threading
from layers import *
from optimizers import * 
from utils import *
from sklearn.cluster import KMeans

class Cluster:
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
        self.cluster_center = None
        self.belongs = []
    
    def fit(self, data):
        self.initial_cluster_center(data)
        distance_matrix = cp.array([
            cp.sum(cp.square(data - self.cluster_center[i]), axis=-1) for i in range(self.cluster_num)
        ])
        this_belongs = cp.argmin(distance_matrix, axis=0)
        last_belongs = cp.zeros_like(this_belongs)
        
        def if_two_array_equal(arr1, arr2, softmargin=0):
            return True if cp.sum(arr1 == arr2) <= softmargin else False

        while not if_two_array_equal(this_belongs, last_belongs, 0.05 * len(data)):
            last_belongs = this_belongs.copy()
            
            self.cluster_center = cp.array([cp.mean(data[cp.where(last_belongs == i)], axis=0) for i in range(self.cluster_num)])
            distance_matrix = cp.array([
                cp.sum(cp.square(data - self.cluster_center[i]), axis=-1) for i in range(self.cluster_num)
            ])
            this_belongs = cp.argmin(distance_matrix, axis=0)
        
        self.belongs = this_belongs.copy()
        self.cluster_centers_ = self.cluster_center.copy()
        print('Finished clustering!')

    def initial_cluster_center(self, data):
        cluster_center = cp.array(data[cp.random.choice(len(data), 1)])
        k = 1
        vis = [0] * len(data)
        while k < self.cluster_num:
            distance = cp.sum(cp.abs(data - cp.mean(cluster_center, axis=0)), axis=-1)
            i = 0
            while True:
                i -= 1
                next_cluster_center_index = int(cp.argsort(distance)[i])
                # print(next_cluster_center_index)

                if vis[next_cluster_center_index] == 0:
                    vis[next_cluster_center_index] = 1
                    break 
            cluster_center = cp.concatenate([cluster_center, data[next_cluster_center_index].reshape((1, ) + data[next_cluster_center_index].shape)], axis=0)
            k += 1

        self.cluster_center = cluster_center

def grads_clip(grads, clipvalue=1.):
    grads[cp.where(cp.abs(grads) > clipvalue)] = clipvalue
    return grads

class SGD:
    def __init__(self, steps_per_epoch, all_epochs=None, cycles=4, lr=0.01, momentum=0., clipvalue=1., decay=1e-8):
        self.lr = lr 
        self.momentum = momentum 
        self.init_flag = True
        self.v = None
        self.decay = decay
        self.clipvalue = clipvalue
        self.steps_per_epoch = steps_per_epoch
        self.steps = 0
        self.all_epochs = all_epochs
        self.cycles = cycles
    
    def process(self, params, grads):
        if self.init_flag:
            self.v = {}
            for key, value in params.items():
                self.v[key] = cp.zeros_like(value)
            self.init_flag = False 

        for key in params.keys():
            grads[key] = grads_clip(grads[key], clipvalue=1.)
            self.v[key] = self.momentum * self.v[key] - self.lr_schedule() * grads[key]
            params[key] += self.v[key]
            self.steps += 1
    
    def lr_schedule(self, Flag=True): 
        if Flag:
            lr_ = cosine_annealing(self.all_epochs, self.steps // self.steps_per_epoch, self.cycles, self.lr)
        else:
            lr_ = self.lr
        return lr_

def cosine_annealing(epochs, cur_epoch, cycles, base_lr):
    epoch_per_cycle = epochs // cycles
    cos_inner = (cp.pi * (cur_epoch % epoch_per_cycle)) / epoch_per_cycle
    return base_lr / 2 * (1 + cp.cos(cos_inner))

    
class RBF:
    def __init__(self, hidden_nodes):
        self.hidden_nodes = hidden_nodes
        self.cluster = Cluster(hidden_nodes)
        self.rbf_nodes = []
        self.layers = {}
        self.params = {}
        self.active_values = None
        self.output = None
        self.batch_size = 208
        self.optimizer = SGD(208, lr=0.01, momentum=0.5)
    
    def build_model(self):
        self.params['W1'] = GaussionInit((self.hidden_nodes, 64), self.hidden_nodes)
        self.params['b1'] = cp.zeros((64,))
        self.layers['affine_1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['relu'] = ReLU()
        self.params['W2'] = GaussionInit((64, 2), 64)
        self.params['b2'] = cp.zeros((2,))
        self.layers['affine_2'] = Affine(self.params['W2'], self.params['b2'])

        self.output = SoftmaxLossLayer()

    def fit(self, X, Y, epochs=1000):
        self.build_model()
        self.cluster.fit(X)
        self.optimizer.all_epochs = epochs
        cmax = cp.array([[cp.max(cp.dot((self.cluster.cluster_centers_ - c).T, (self.cluster.cluster_centers_ - c))), c] for c in self.cluster.cluster_centers_])
        var = cmax[:, 0] / cp.sqrt(2. * self.hidden_nodes)
        self.rbf_nodes = [activation(cmax[i, 1], var[i]) for i in range(self.hidden_nodes)]
        self.predict(X)
        for i in range(epochs):
            x = self.layers['affine_1'].forward(self.active_values)
            x = self.layers['relu'].forward(x)
            x = self.layers['affine_2'].forward(x)
            self.output.forward(x, Y)
            grads = {}
            dout = self.output.backward()
            dout = self.layers['affine_2'].backward(dout)
            dout = self.layers['relu'].backward(dout)
            dout = self.layers['affine_1'].backward(dout)
            grads['W1'] = self.layers['affine_1'].dW 
            grads['b1'] = self.layers['affine_1'].db
            grads['W2'] = self.layers['affine_2'].dW 
            grads['b2'] = self.layers['affine_2'].db
            self.optimizer.process(self.params, grads)

    def predict(self, X):
        active_values = []
        for rbf in self.rbf_nodes:
            active_values.append(rbf.forward(X))
        self.active_values = cp.array(active_values)
        self.active_values = self.active_values.T
        x = self.layers['affine_1'].forward(self.active_values)
        x = self.layers['relu'].forward(x)
        x = self.layers['affine_2'].forward(x)
        return x
        
def GaussionInit(shape, num):
    return cp.random.rand(*shape) * cp.sqrt(1. / num)     

class activation:
    def __init__(self, c, var):
        self.c = c
        self.var = var
    
    def forward(self, x):
        bias = (x - self.c)
        radius = cp.sum(bias * bias, axis=1)
        active_value = cp.exp(-1. / (2. * self.var * self.var + 1e-7) * radius)
        return active_value


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