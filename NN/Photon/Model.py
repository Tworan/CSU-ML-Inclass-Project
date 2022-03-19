import os, sys 
sys.path.append('/yourpath/NN/Photon')
from layers import * 
from optimizers import *  
from utils import *
from collections import OrderedDict
import pickle 

class ConvNet:
    def __init__(self, datasets='Mnist', opt='sgd'): 
        if datasets.lower() == 'mnist':
            self.channels = 1
            self.size = 28
            self.data_n = 60000
            self.last_feature_size = 1
        elif datasets.lower() == 'cifar_10':
            self.channels = 3
            self.size = 32
            self.data_n = 50000
            self.last_feature_size = 2
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
        Conv2d 64       1
        BN              2
        ReLU            3
        Conv2d 64       4
        BN              5
        ReLU            6
        Maxpooling2D    7

        Conv2d 128      8
        BN              9
        ReLU            10
        Conv2d 128      11
        BN              12
        ReLU            13
        Maxpooling2D    14
        
        Conv2d 256      15
        BN              16
        ReLU            17
        Conv2d 256      18
        BN              19
        ReLU            20
        Maxpooling2D    21

        Conv2d 512      22
        BN              23
        ReLU            24
        conv2d 512      25
        BN              26
        ReLU            27
        Maxpooling2D    28

        Dropout 0.3     29
        Dense 256       30
        BN              31
        ReLU            32
        Dropout 0.3     33
        Dense 256       34
        BN              35
        ReLU            36
        Dense 10 classification softmax crossentropy
        '''
        self.paramized_layers_name = {
            'block1_conv1': 1,
            # 'bn_1': 2, 
            'block1_conv2': 4,
            # 'bn_2': 5,
            'block2_conv1': 8, 
            # 'bn_3': 9,
            'block2_conv2': 11,
            # 'bn_4': 12,
            'block3_conv1': 15, 
            # 'bn_5': 16,
            'block3_conv2': 18, 
            # 'bn_6': 19,
            'block4_conv1': 22, 
            # 'bn_7': 23,
            'block4_conv2': 25,
            # 'bn_8': 26,
            'affine_1': 30, 
            # 'bn_9': 31,
            'affine_2': 34,
            # 'bn_10': 35,
            'affine_3': 37
            }
        self.layers = OrderedDict()
        self.params = {}
        #* ============================================================Block1 32
        self.params['W_block1_conv1'] = self.GaussionInit((64, 3, 3, self.channels), self.size * self.size * self.channels)
        self.params['b_block1_conv1'] = cp.zeros((64))
        
        self.layers['block1_conv1'] = Conv2D(self.params['W_block1_conv1'], self.params['b_block1_conv1'], padding='same')
        self.layers['bn_1'] = BatchNormalization(1, 0)
        self.layers['ReLU_1'] = ReLU()
    
        self.params['W_block1_conv2'] = self.GaussionInit((64, 3, 3, 64), 64 * 3 * 3)
        self.params['b_block1_conv2'] = cp.zeros((64))
        self.layers['block1_conv2'] = Conv2D(self.params['W_block1_conv2'], self.params['b_block1_conv2'], padding='same')
        self.layers['bn_2'] = BatchNormalization(1, 0)
        self.layers['ReLU_2'] = ReLU()
    
        self.layers['maxpooling1'] = MaxPooling2D((2, 2), padding='same')
        #* ============================================================Block2 16
        self.params['W_block2_conv1'] = self.GaussionInit((128, 3, 3, 64), 64 * 3 * 3)
        self.params['b_block2_conv1'] = cp.zeros((128))
        self.layers['block2_conv1'] = Conv2D(self.params['W_block2_conv1'], self.params['b_block2_conv1'], padding='same')
        self.layers['bn_3'] = BatchNormalization(1, 0)
        self.layers['ReLU_3'] = ReLU()
        self.params['W_block2_conv2'] = self.GaussionInit((128, 3, 3, 128), 128 * 3 * 3)
        self.params['b_block2_conv2'] = cp.zeros((128))
        self.layers['block2_conv2'] = Conv2D(self.params['W_block2_conv2'], self.params['b_block2_conv2'], padding='same')
        self.layers['bn_4'] = BatchNormalization(1, 0)
        self.layers['ReLU_4'] = ReLU()
        self.layers['maxpooling2'] = MaxPooling2D((2, 2), padding='same')
        #* ============================================================Block3 8
        self.params['W_block3_conv1'] = self.GaussionInit((256, 3, 3, 128), 128 * 3 * 3)
        self.params['b_block3_conv1'] = cp.zeros((256))
        self.layers['block3_conv1'] = Conv2D(self.params['W_block3_conv1'], self.params['b_block3_conv1'], padding='same')
        self.layers['bn_5'] = BatchNormalization(1, 0)
        self.layers['ReLU_5'] = ReLU()
        self.params['W_block3_conv2'] = self.GaussionInit((256, 3, 3, 256), 256 * 3 * 3)
        self.params['b_block3_conv2'] = cp.zeros((256))
        self.layers['block3_conv2'] = Conv2D(self.params['W_block3_conv2'], self.params['b_block3_conv2'], padding='same')
        self.layers['bn_6'] = BatchNormalization(1, 0)
        self.layers['ReLU_6'] = ReLU()
        self.layers['maxpooling3'] = MaxPooling2D((2, 2), padding='same')
        #* ============================================================Block4 4
        self.params['W_block4_conv1'] = self.GaussionInit((512, 3, 3, 256), 256 * 3 * 3)
        self.params['b_block4_conv1'] = cp.zeros((512))
        self.layers['block4_conv1'] = Conv2D(self.params['W_block4_conv1'], self.params['b_block4_conv1'], padding='same')
        self.layers['bn_7'] = BatchNormalization(1, 0)
        self.layers['ReLU_7'] = ReLU()
        self.params['W_block4_conv2'] = self.GaussionInit((512, 3, 3, 512), 512 * 3 * 3)
        self.params['b_block4_conv2'] = cp.zeros((512))
        self.layers['block4_conv2'] = Conv2D(self.params['W_block4_conv2'], self.params['b_block4_conv2'], padding='same')
        self.layers['bn_8'] = BatchNormalization(1, 0)
        self.layers['ReLU_8'] = ReLU()
        self.layers['maxpooling4'] = MaxPooling2D((2, 2), padding='same')
        #* ============================================================Classification 2
        self.layers['dropout_1'] = Dropout(0.3)
        self.params['W_affine_1'] = self.GaussionInit((512 * self.last_feature_size * self.last_feature_size, 256), 512 * 3 * 3)
        self.params['b_affine_1'] = cp.zeros((256,))
        self.layers['affine_1'] = Affine(self.params['W_affine_1'], self.params['b_affine_1'])
        self.layers['bn_9'] = BatchNormalization(1, 0)
        self.layers['ReLU_9'] = ReLU()
        self.params['W_affine_2'] = self.GaussionInit((256, 256), 512)
        self.params['b_affine_2'] = cp.zeros((256,)) 
        self.layers['dropout_2'] = Dropout(0.3)
        self.layers['affine_2'] = Affine(self.params['W_affine_2'], self.params['b_affine_2'])
        self.layers['bn_10'] = BatchNormalization(1, 0)
        self.layers['ReLU_10'] = ReLU()
        self.params['W_affine_3'] = self.GaussionInit((256, 10), 256)
        self.params['b_affine_3'] = cp.zeros((10,))
        self.layers['affine_3'] = Affine(self.params['W_affine_3'], self.params['b_affine_3'])
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
