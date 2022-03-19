import numpy as np 
import cupy as cp 
def grads_clip(grads, clipvalue=1.):
    grads /= cp.sum(grads * grads) * clipvalue
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
                self.v[key] = np.zeros_like(value)
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

class RMSprop:
    def __init__(self, steps_per_epoch, all_epochs=None, cycles=4, lr=0.001, p=0.9, momentum=0., clipvalue=1., decay=1e-8):
        self.lr = lr  
        self.momentum = momentum
        self.clipvalue = clipvalue 
        self.decay = decay 
        self.steps_per_epoch = steps_per_epoch
        self.r = None 
        self.v = None
        self.p = p
        self.steps = 0
        self.all_epochs = all_epochs
        self.cycles = cycles
        self.init_flag = True

    
    def process(self, params, grads):
        if self.init_flag:
            self.v = {}
            self.r = {} 
            for key, value in params.items():
                self.v[key] = np.zeros_like(value) 
                self.r[key] = np.zeros_like(value)
            self.init_flag = False

        for key in params.keys():
            grads[key] = grads_clip(grads[key], clipvalue=1.)
            self.r[key] = self.p * self.r[key] + (1 - self.p) * grads[key] * grads[key] 
            self.v[key] = self.momentum * self.v[key] - self.lr_schedule() / (np.sqrt(self.r[key]) + 1e-7) * grads[key]
            params[key] += self.v[key] 
            self.steps += 1

    def lr_schedule(self, Flag=True): 
        if Flag:
            lr_ = cosine_annealing(self.all_epochs, self.steps // self.steps_per_epoch, self.cycles, self.lr)
        else:
            lr_ = self.lr
        return lr_
    
class Adam:
    def __init__(self, steps_per_epoch, all_epochs=None, cycles=4, lr=0.001, clipvalue=1., beta_1=0.9, beta_2=0.999, decay=1e-8, momentum=0.):
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.clipvalue = clipvalue
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.steps = 0
        self.all_epochs = all_epochs
        self.cycles = cycles
        self.init_flag = True 
        self.s = None 
        self.r = None 
    
    def process(self, params, grads): 
        if self.init_flag:
            self.s = {}
            self.r = {} 
            for key, value in params.items(): 
                self.s[key] = np.zeros_like(value)
                self.r[key] = np.zeros_like(value)
            self.init_flag = False 
        
        self.steps += 1
        for key in params.keys():
            grads[key] = grads_clip(grads[key], clipvalue=1.)
            self.s[key] = (self.beta_1 * self.s[key] + (1. - self.beta_1) * grads[key]) / (1. - self.beta_1**self.steps)
            self.r[key] = (self.beta_2 * self.r[key] + (1. - self.beta_2) * grads[key] * grads[key]) / (1. - self.beta_2**self.steps + 1e-7)
            params[key] = params[key] - self.lr_schedule() * self.s[key] / (np.sqrt(self.r[key]) + 1e-7)
            
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

    