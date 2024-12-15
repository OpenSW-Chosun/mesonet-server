# -*- coding:utf-8 -*-

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation  # Activation 추가

IMGWIDTH = 256

class Classifier:
    def __init__(self):
        self.model = 0
    
    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)


class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    
    def init_model(self): 
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        
        # 첫 번째 Convolutional Block
        x1 = Conv2D(8, (3, 3), padding='same')(x)  # 필터 개수 유지
        x1 = BatchNormalization()(x1)
        x1 = Activation('swish')(x1)  # 활성화 함수 변경 (ReLU → Swish)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        # 두 번째 Convolutional Block
        x2 = Conv2D(8, (5, 5), padding='same')(x1)  # 필터 개수 유지
        x2 = BatchNormalization()(x2)
        x2 = Activation('swish')(x2)  # 활성화 함수 변경
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        # 세 번째 Convolutional Block
        x3 = Conv2D(16, (5, 5), padding='same')(x2)  # 필터 개수 유지
        x3 = BatchNormalization()(x3)
        x3 = Activation('swish')(x3)  # 활성화 함수 변경
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        # 네 번째 Convolutional Block
        x4 = Conv2D(16, (5, 5), padding='same')(x3)  # 필터 개수 유지
        x4 = BatchNormalization()(x4)
        x4 = Activation('swish')(x4)  # 활성화 함수 변경
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        # Fully Connected Layers
        y = Flatten()(x4)
        y = Dropout(0.4)(y)  # Dropout 비율 조정 (0.3 → 0.4)
        y = Dense(16)(y)  # 노드 개수 유지
        y = LeakyReLU(alpha=0.1)(y)  # LeakyReLU 유지
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)  # 최종 출력층

        return KerasModel(inputs=x, outputs=y)


