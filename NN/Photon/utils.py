import cupy as cp 
import numpy as np 
def to_GPU(x):
    return cp.array(x) 

def to_CPU(x): 
    return cp.asnumpy(x)
    
# 这两个应该在CPU上运行
def img2col(x, Filter_H, Filter_W, strides, out_H, out_W):
    '''
    用法：输入图片，将图片拍平为适合卷积核批操作的方式
    '''
    # x = to_CPU(x)
    Sample_num, H, W, Channel = x.shape 
    pad_size = (int(Filter_H // 2), int(Filter_W // 2))
    img = cp.pad(x, [(0, 0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0)], 'constant')
    col = cp.zeros((Sample_num, Filter_H, Filter_W, out_H, out_W, Channel))

    for h in range(Filter_H):
        end_h = h + out_W * strides[0]
        for w in range(Filter_W):
            end_w = w + out_H * strides[1] 
            # Samples Y X H W Channels
            col[:, w, h, :, :, :] = img[:, h: end_h: strides[0], w: end_w: strides[1], :]

    return col.transpose([0, 3, 4, 1, 2, 5]).reshape(Sample_num * out_H * out_W, -1)

def col2img(col, x_shape, Filter_H, Filter_W, strides, out_H, out_W):
    '''
    用法： 将输入的图片还原回去
    '''
    Sample_num, H, W, Channel = x_shape 
    pad_size = (int(Filter_H // 2), int(Filter_W // 2))
    col = col.reshape(Sample_num, out_H, out_W, Filter_H, Filter_W, Channel).transpose(0, 3, 4, 1, 2, 5)
    img = cp.zeros((Sample_num, pad_size[0] * 2 + H, pad_size[1] * 2 + W, Channel))
    for h in range(Filter_H):
        end_h = h + out_W * strides[0]
        for w in range(Filter_W):
            end_w = w + out_H * strides[1] 
            # Samples Y X H W Channels
            img[:, h: end_h: strides[0], w: end_w: strides[1], :] = col[:, w, h, :, :, :]
    return img[:, pad_size[0]: H + pad_size[0], pad_size[1]: W + pad_size[1], :]