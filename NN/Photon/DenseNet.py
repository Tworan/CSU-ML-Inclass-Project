import os, sys 
sys.path.append('/yourpath/NN/Photon')
from layers import * 
from optimizers import *  
from utils import *
from collections import OrderedDict
import pickle 

class DenseNet:
    def __init__(self, datasets='mnist', opt='SGD'): 
        if datasets.lower() == 'mnist':
            self.channels = 1
            self.size = 28
            self.data_n = 60000
            self.last_feature_size = 1
        elif datasets.lower() == 'cifar_10':
            self.channels = 3
            self.size = 32
            self.data_n = 50000
            self.last_feature_size = 1
        self.datasets = datasets
        self.model = self.build_model()
        self.batch_size = 64
        if opt.lower() == 'sgd':
            self.optimizer = SGD(self.data_n // self.batch_size, lr=0.01, momentum=0.8)
        else: 
            self.optimizer = RMSprop(self.data_n // self.batch_size, lr=0.001, momentum=0.8)
        #* datas[0] -> X datas[1] -> Y
        self.datas = None 
        self.evaluate_datas = None
        self.history = []


    def train(self, epochs=20):
        self.optimizer.all_epochs = epochs
        for epoch in range(epochs):
            losses = 0
            for steps in range(len(self.datas[0]) // self.batch_size):
                grads, loss = self.cal_gradient(to_GPU(self.datas[0][steps * self.batch_size: (steps + 1) * self.batch_size]), to_GPU(self.datas[1][steps * self.batch_size: (steps + 1) * self.batch_size]))
                self.fit(grads)
                if steps % 10 == 0:
                    print('Epoch: {} Step: {} Loss: {}'.format(epoch, steps+1, cp.mean(loss)))
                    self.history.append(np.mean(to_CPU(loss)))
    
    def score(self, X_test, Y_test):
        batch_size = 32 
        if self.datasets.lower() == 'mnist':
            X_test = np.expand_dims(X_test, axis=-1) / 255.
        else:
            X_test = X_test / 255.

        pred = np.empty((1, 10))
        for i in range(len(X_test) // batch_size):
            datas = to_GPU(X_test[i*batch_size: (i+1)*batch_size])
            _pred = to_CPU(self.predict(datas))
            pred = np.concatenate([pred, _pred], axis=0)
        score = np.mean(np.argmax(pred[1:], axis=1) == Y_test[:len(pred[1:])].flatten()) 
        print('Score is {}'.format(score)) 
        return score, pred[1:]

    def call_evaluate(self):
        if not self.evaluate_datas:
            print('Evaluate data is empty!')
            return 
        print('Score: {}'.format(np.mean(np.argmax(to_CPU(self.predict(to_GPU(self.evaluate_datas[0]), False)), axis=1) == np.argmax(self.evaluate_datas[1], axis=1))))

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x.copy())
        return x 
    
    def cal_loss(self, x, y):
        y_out = self.predict(x)
        loss = self.output.forward(y_out, y)
        return loss 

    def cal_gradient(self, x, y):
        loss = self.cal_loss(x, y)

        dout = self.output.backward()
        for layer in list(self.layers.values())[::-1]:
            dout = layer.backward(dout) 
        
        grads = {}
        for name, value in self.paramized_layers_name.items():
            grads['W_{}'.format(name)] = self.layers[list(self.layers.keys())[value-1]].dW 
            grads['b_{}'.format(name)] = self.layers[list(self.layers.keys())[value-1]].db 

        return grads, loss
    
    def fit(self, grads):
        self.optimizer.process(self.params, grads)


    def build_model(self):
        '''
        网络架构
        input    28 * 28 
        affine_1 512        1
        bn_1                2
        ReLU                3
        dropout_1           4

        affine_2 256        5
        bn_2                6
        ReLU                7
        dropout_2           8

        affine_3 256        9
        bn_3                10
        ReLU                11
        dropout_3           12

        affine_4 128        13
        bn_4                14
        ReLU                15
        dropout_4           16

        affine_5 64         17
        bn_5                18
        ReLU                19
        Dense 10 classification softmax crossentropy
        '''
        self.paramized_layers_name = {
            'affine_1': 1,
            'affine_2': 5,
            'affine_3': 9,
            'affine_4': 13,
            'affine_5': 17,
            'affine_6': 20
            }
        self.layers = OrderedDict()
        self.params = {}
        #* =====================================================
        self.params['W_affine_1'] = self.GaussionInit((self.size * self.size * self.channels, 512), self.size * self.size * self.channels)
        self.params['b_affine_1'] = cp.zeros((512,))
        self.layers['affine_1'] = Affine(self.params['W_affine_1'], self.params['b_affine_1'])
        self.layers['bn_1'] = BatchNormalization(1, 0)
        self.layers['ReLU_1'] = ReLU()

        #* =====================================================
        self.params['W_affine_2'] = self.GaussionInit((512, 256), 512)
        self.params['b_affine_2'] = cp.zeros((256,)) 
        self.layers['dropout_2'] = Dropout(0.3)
        self.layers['affine_2'] = Affine(self.params['W_affine_2'], self.params['b_affine_2'])
        self.layers['bn_2'] = BatchNormalization(1, 0)
        self.layers['ReLU_2'] = ReLU()

        #* =====================================================
        self.params['W_affine_3'] = self.GaussionInit((256, 256), 256)
        self.params['b_affine_3'] = cp.zeros((256,)) 
        self.layers['dropout_3'] = Dropout(0.3)
        self.layers['affine_3'] = Affine(self.params['W_affine_3'], self.params['b_affine_3'])
        self.layers['bn_3'] = BatchNormalization(1, 0)
        self.layers['ReLU_3'] = ReLU()

        #* =====================================================
        self.params['W_affine_4'] = self.GaussionInit((256, 128), 256)
        self.params['b_affine_4'] = cp.zeros((128,)) 
        self.layers['dropout_4'] = Dropout(0.3)
        self.layers['affine_4'] = Affine(self.params['W_affine_4'], self.params['b_affine_4'])
        self.layers['bn_4'] = BatchNormalization(1, 0)
        self.layers['ReLU_4'] = ReLU()

        #* =====================================================
        self.params['W_affine_5'] = self.GaussionInit((128, 64), 128)
        self.params['b_affine_5'] = cp.zeros((64,)) 
        self.layers['dropout_5'] = Dropout(0.3)
        self.layers['affine_5'] = Affine(self.params['W_affine_5'], self.params['b_affine_5'])
        self.layers['bn_5'] = BatchNormalization(1, 0)
        self.layers['ReLU_5'] = ReLU()

        #* =====================================================
        self.params['W_affine_6'] = self.GaussionInit((64, 10), 64)
        self.params['b_affine_6'] = cp.zeros((10,))
        self.layers['affine_6'] = Affine(self.params['W_affine_6'], self.params['b_affine_6'])
        self.output = SoftmaxLossLayer()
    
    def XavierInit(self, shape):
        return cp.random.rand(*shape) 

    def GaussionInit(self, shape, nodes_num): 
        return cp.random.rand(*shape) * cp.sqrt(2. / nodes_num) * 0.01

    def save_model(self, save_path):
        json_formated_data = pickle.dumps(self.params)
        with open(save_path, 'wb') as f:
            f.write(json_formated_data)
    
    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            params = pickle.load(f)
        for key, value in params.items():
            self.params[key] = value 
        for key, value in self.params.items():
            if key[0] == 'W':
                self.layers[key[2:]].W = value
            elif key[0] == 'b':
                self.layers[key[2:]].b = value