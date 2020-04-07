from keras.models import Model as M
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from keras.applications import *
import numpy as np
import keras
from keras import *

class Model:
    def __init__(self, input_dim, classes, lr=0.001, epochs=1):
        self.input_dim = input_dim
        self.num_classes = classes
        self.learning_rate = lr
        self.epochs = epochs

    def lr_schedule(self, epoch):
        if epoch % 5 == 0:
            self.learning_rate = self.learning_rate/10
        print('Learning rate: ', self.learning_rate)
        return self.learning_rate

    def model(self):
        premodel = InceptionResNetV2(weights = "imagenet", include_top=False, input_shape = self.input_dim)
        layer = premodel.output
        layer = Flatten()(layer)
        layer = Dense(128, kernel_initializer='he_normal', activation='relu')(layer)
        layer = Dense(16, kernel_initializer='he_normal', activation='relu')(layer)
        predictions = Dense(self.num_classes, activation='softmax')(layer)

        # creating the final model 
        model = M(input = premodel.input, output = predictions)
        model.compile(optimizer=Adam(lr=self.lr_schedule(1)),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        
        return model

