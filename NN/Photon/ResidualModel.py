import os, sys 
sys.path.append('/yourpath/NN/Photon')
from layers import * 
from optimizers import *  
from utils import *
from collections import OrderedDict
import json, pickle

class ResNet:
    def __init__(self, datasets='mnist', opt='sgd'): 
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
        get_layer = lambda index: self.layers[list(self.layers.keys())[index-1]]
        x_1, x_2 = get_layer(1).forward(x)
        x_1 = get_layer(2).forward(x_1)
        x_1 = get_layer(3).forward(x_1)
        x_1 = get_layer(4).forward(x_1)
        x_1 = get_layer(5).forward(x_1)

        x_2 = get_layer(6).forward(x_2)
        x = get_layer(7).forward(x_1, x_2)
        x = get_layer(8).forward(x)
        x = get_layer(9).forward(x)
        x = get_layer(10).forward(x)

        x_1, x_2 = get_layer(11).forward(x)
        x_1 = get_layer(12).forward(x_1)
        x_1 = get_layer(13).forward(x_1)
        x_1 = get_layer(14).forward(x_1)
        x_1 = get_layer(15).forward(x_1)

        x_2 = get_layer(16).forward(x_2)
        x = get_layer(17).forward(x_1, x_2)
        x = get_layer(18).forward(x)
        x = get_layer(19).forward(x)
        x = get_layer(20).forward(x)

        x_1, x_2 = get_layer(21).forward(x)
        x_1 = get_layer(22).forward(x_1)
        x_1 = get_layer(23).forward(x_1)
        x_1 = get_layer(24).forward(x_1)
        x_1 = get_layer(25).forward(x_1)

        x_2 = get_layer(26).forward(x_2)
        x = get_layer(27).forward(x_1, x_2)
        x = get_layer(28).forward(x)
        x = get_layer(29).forward(x)
        x = get_layer(30).forward(x)

        x_1, x_2 = get_layer(31).forward(x)
        x_1 = get_layer(32).forward(x_1)
        x_1 = get_layer(33).forward(x_1)
        x_1 = get_layer(34).forward(x_1)
        x_1 = get_layer(35).forward(x_1)

        x_2 = get_layer(36).forward(x_2)
        x = get_layer(37).forward(x_1, x_2)
        x = get_layer(38).forward(x)
        x = get_layer(39).forward(x)
        x = get_layer(40).forward(x)

        x = get_layer(41).forward(x)
        x = get_layer(42).forward(x)
        x = get_layer(43).forward(x)
        x = get_layer(44).forward(x)
        x = get_layer(45).forward(x)
        x = get_layer(46).forward(x)
        x = get_layer(47).forward(x)
        x = get_layer(48).forward(x)
        x = get_layer(49).forward(x)
        return x 
    
    def cal_loss(self, x, y):
        y_out = self.predict(x)
        loss = self.output.forward(y_out, y)
        return loss 

    def cal_gradient(self, x, y):
        loss = self.cal_loss(x, y)
        get_layer = lambda index: self.layers[list(self.layers.keys())[index-1]]

        dout = self.output.backward()
        dout = get_layer(49).backward(dout)
        dout = get_layer(48).backward(dout)
        dout = get_layer(47).backward(dout)
        dout = get_layer(46).backward(dout)
        dout = get_layer(45).backward(dout)
        dout = get_layer(44).backward(dout)
        dout = get_layer(43).backward(dout)
        dout = get_layer(42).backward(dout)
        dout = get_layer(41).backward(dout)
        dout = get_layer(40).backward(dout)
        dout = get_layer(39).backward(dout)
        dout = get_layer(38).backward(dout)

        dout_1, dout_2 = get_layer(37).backward(dout)
        dout_2 = get_layer(36).backward(dout_2)
        dout_1 = get_layer(35).backward(dout)
        dout_1 = get_layer(34).backward(dout_1)
        dout_1 = get_layer(33).backward(dout_1)
        dout_1 = get_layer(32).backward(dout_1)
        dout = get_layer(31).backward(dout_1, dout_2)

        dout = get_layer(30).backward(dout)
        dout = get_layer(29).backward(dout) 
        dout = get_layer(28).backward(dout)
        dout_1, dout_2 = get_layer(27).backward(dout)
        dout_2 = get_layer(26).backward(dout_2)
        dout_1 = get_layer(25).backward(dout_1)
        dout_1 = get_layer(24).backward(dout_1)
        dout_1 = get_layer(23).backward(dout_1)
        dout_1 = get_layer(22).backward(dout_1)
        dout = get_layer(21).backward(dout_1, dout_2)

        dout = get_layer(20).backward(dout)
        dout = get_layer(19).backward(dout) 
        dout = get_layer(18).backward(dout)
        dout_1, dout_2 = get_layer(17).backward(dout)
        dout_2 = get_layer(16).backward(dout_2)
        dout_1 = get_layer(15).backward(dout_1)
        dout_1 = get_layer(14).backward(dout_1)
        dout_1 = get_layer(13).backward(dout_1)
        dout_1 = get_layer(12).backward(dout_1)
        dout = get_layer(11).backward(dout_1, dout_2)

        dout = get_layer(10).backward(dout)
        dout = get_layer(9).backward(dout) 
        dout = get_layer(8).backward(dout)
        dout_1, dout_2 = get_layer(7).backward(dout)
        dout_2 = get_layer(6).backward(dout_2)
        dout_1 = get_layer(5).backward(dout_1)
        dout_1 = get_layer(4).backward(dout_1)
        dout_1 = get_layer(3).backward(dout_1)
        dout_1 = get_layer(2).backward(dout_1)
        dout = get_layer(1).backward(dout_1, dout_2)
        
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
                        Diversion               1
        Conv2d 64       2       
        BN              3       Conv2d(1x1) 64  6
        ReLU            4
        Conv2d 64       5
                        Add                     7
        BN              8
        ReLU            9
        Maxpooling2D    10

                        Diversion               11
        Conv2d 128      12
        BN              13      Conv2d(1x1) 128 16
        ReLU            14
        Conv2d 128      15
                        Add                     17
        BN              18
        ReLU            19
        Maxpooling2D    20

                        Diversion               21 
        Conv2d 256      22
        BN              23      Conv2d(1x1) 256 26
        ReLU            24
        Conv2d 256      25
                        Add                     27         
        BN              28
        ReLU            29
        Maxpooling2D    30      

                        Diversion               31   
        Conv2d 512      32
        BN              33      Conv2d(1x1) 512 36
        ReLU            34
        conv2d 512      35
                        Add                     37
        BN              38
        ReLU            39
        Maxpooling2D    40

        Dropout 0.3     41
        Dense 256       42
        BN              43
        ReLU            44
        Dropout 0.3     45
        Dense 256       46
        BN              47
        ReLU            48
        Dense 10 classification softmax crossentropy
        '''
        self.paramized_layers_name = {
            'block1_conv1': 2,
            'block1_conv2': 5,
            'skip_conv1':   6,
            'block2_conv1': 12, 
            'block2_conv2': 15,
            'skip_conv2':   16,
            'block3_conv1': 22, 
            'block3_conv2': 25, 
            'skip_conv3':   26,
            'block4_conv1': 32, 
            'block4_conv2': 35,
            'skip_conv4':   36,
            'affine_1': 42, 
            'affine_2': 46,
            'affine_3': 49
            }
        self.layers = OrderedDict()
        self.params = {}
        #* ============================================================Block1 32
        self.layers['div_1'] = Diversion()
        
        self.params['W_block1_conv1'] = self.HeInit((64, 3, 3, self.channels), self.size * self.size * self.channels) 
        self.params['b_block1_conv1'] = cp.zeros((64))
        self.layers['block1_conv1'] = Conv2D(self.params['W_block1_conv1'], self.params['b_block1_conv1'], padding='same')
        self.layers['bn_1'] = BatchNormalization(1, 0)
        self.layers['ReLU_1'] = ReLU()
        self.params['W_block1_conv2'] = self.HeInit((64, 3, 3, 64), 64 * 3 * 3)
        self.params['b_block1_conv2'] = cp.zeros((64))
        self.layers['block1_conv2'] = Conv2D(self.params['W_block1_conv2'], self.params['b_block1_conv2'], padding='same')
        
        self.params['W_skip_conv1'] = self.HeInit((64, 3, 3, self.channels), self.size *  self.size * self.channels)
        self.params['b_skip_conv1'] = cp.zeros((64))
        self.layers['skip_conv1'] = Conv2D(self.params['W_skip_conv1'], self.params['b_skip_conv1'], padding='same')

        self.layers['add_1'] = Add()
        self.layers['bn_2'] = BatchNormalization(1, 0)
        self.layers['ReLU_2'] = ReLU()
        self.layers['maxpooling1'] = MaxPooling2D((2, 2), padding='same')
        #* ============================================================Block2 16
        self.layers['div_2'] = Diversion()
        
        self.params['W_block2_conv1'] = self.HeInit((128, 3, 3, 64), 64 * 3 * 3)
        self.params['b_block2_conv1'] = cp.zeros((128))
        self.layers['block2_conv1'] = Conv2D(self.params['W_block2_conv1'], self.params['b_block2_conv1'], padding='same')
        self.layers['bn_3'] = BatchNormalization(1, 0)
        self.layers['ReLU_3'] = ReLU()
        self.params['W_block2_conv2'] = self.HeInit((128, 3, 3, 128), 128 * 3 * 3)
        self.params['b_block2_conv2'] = cp.zeros((128))
        self.layers['block2_conv2'] = Conv2D(self.params['W_block2_conv2'], self.params['b_block2_conv2'], padding='same')
        
        self.params['W_skip_conv2'] = self.HeInit((128, 3, 3, 64), 64 * 3 * 3)
        self.params['b_skip_conv2'] = cp.zeros((128))
        self.layers['skip_conv2'] = Conv2D(self.params['W_skip_conv2'], self.params['b_skip_conv2'], padding='same')

        self.layers['add_2'] = Add()
        self.layers['bn_4'] = BatchNormalization(1, 0)
        self.layers['ReLU_4'] = ReLU()
        self.layers['maxpooling2'] = MaxPooling2D((2, 2), padding='same')
        #* ============================================================Block3 8
        self.layers['div_3'] = Diversion()
        
        self.params['W_block3_conv1'] = self.HeInit((256, 3, 3, 128), 128 * 3 * 3)
        self.params['b_block3_conv1'] = cp.zeros((256))
        self.layers['block3_conv1'] = Conv2D(self.params['W_block3_conv1'], self.params['b_block3_conv1'], padding='same')
        self.layers['bn_5'] = BatchNormalization(1, 0)
        self.layers['ReLU_5'] = ReLU()
        self.params['W_block3_conv2'] = self.HeInit((256, 3, 3, 256), 256 * 3 * 3)
        self.params['b_block3_conv2'] = cp.zeros((256))
        self.layers['block3_conv2'] = Conv2D(self.params['W_block3_conv2'], self.params['b_block3_conv2'], padding='same')
        
        self.params['W_skip_conv3'] = self.HeInit((256, 3, 3, 128), 128 * 3 * 3)
        self.params['b_skip_conv3'] = cp.zeros((256))
        self.layers['skip_conv3'] = Conv2D(self.params['W_skip_conv3'], self.params['b_skip_conv3'], padding='same')

        self.layers['add_3'] = Add()
        self.layers['bn_6'] = BatchNormalization(1, 0)
        self.layers['ReLU_6'] = ReLU()
        self.layers['maxpooling3'] = MaxPooling2D((2, 2), padding='same')
        #* ============================================================Block4 4
        self.layers['div_4'] = Diversion()
        
        self.params['W_block4_conv1'] = self.HeInit((512, 3, 3, 256), 256 * 3 * 3)
        self.params['b_block4_conv1'] = cp.zeros((512))
        self.layers['block4_conv1'] = Conv2D(self.params['W_block4_conv1'], self.params['b_block4_conv1'], padding='same')
        self.layers['bn_7'] = BatchNormalization(1, 0)
        self.layers['ReLU_7'] = ReLU()
        self.params['W_block4_conv2'] = self.HeInit((512, 3, 3, 512), 512 * 3 * 3)
        self.params['b_block4_conv2'] = cp.zeros((512))
        self.layers['block4_conv2'] = Conv2D(self.params['W_block4_conv2'], self.params['b_block4_conv2'], padding='same')
        
        self.params['W_skip_conv4'] = self.HeInit((512, 3, 3, 256), 256 * 3 * 3)
        self.params['b_skip_conv4'] = cp.zeros((512))
        self.layers['skip_conv4'] = Conv2D(self.params['W_skip_conv4'], self.params['b_skip_conv4'], padding='same')

        self.layers['add_4'] = Add()
        self.layers['bn_8'] = BatchNormalization(1, 0)
        self.layers['ReLU_8'] = ReLU()
        self.layers['maxpooling4'] = MaxPooling2D((2, 2), padding='same')
        #* ============================================================Classification 2
        self.layers['dropout_1'] = Dropout(0.3)
        self.params['W_affine_1'] = self.HeInit((512 * self.last_feature_size * self.last_feature_size, 256), 512 * 3 * 3)
        self.params['b_affine_1'] = cp.zeros((256,))
        self.layers['affine_1'] = Affine(self.params['W_affine_1'], self.params['b_affine_1'])
        self.layers['bn_9'] = BatchNormalization(1, 0)
        self.layers['ReLU_9'] = ReLU()
        self.params['W_affine_2'] = self.HeInit((256, 256), 512)
        self.params['b_affine_2'] = cp.zeros((256,)) 
        self.layers['dropout_2'] = Dropout(0.3)
        self.layers['affine_2'] = Affine(self.params['W_affine_2'], self.params['b_affine_2'])
        self.layers['bn_10'] = BatchNormalization(1, 0)
        self.layers['ReLU_10'] = ReLU()
        self.params['W_affine_3'] = self.HeInit((256, 10), 256)
        self.params['b_affine_3'] = cp.zeros((10,))
        self.layers['affine_3'] = Affine(self.params['W_affine_3'], self.params['b_affine_3'])
        self.output = SoftmaxLossLayer()
    
    def XavierInit(self, shape):
        return cp.random.rand(*shape) 

    def HeInit(self, shape, nodes_num): 
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
        

        
