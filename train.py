import keras
from sklearn.utils import shuffle
from keras.models import Model as M
from keras.layers import *
from keras.optimizers import *
from keras.initializers import *
from keras.applications import *
from keras.preprocessing.image import *
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.regularizers import l1, l2
import sys
from model import Model
from data import Data

trainData = sys.argv[1]
modelFile = sys.argv[2]

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_CLASSES = 10
IMG_DIM = (256, 256, 3)

dataObj = Data(BATCH_SIZE)
trainDataGen, valDataGen = dataObj.loadTrainData(trainData)

model = Model(IMG_DIM, NUM_CLASSES, LEARNING_RATE, EPOCHS)

lrScheduler = LearningRateScheduler(model.lr_schedule)
earlyStopping = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=11)
modelCheckpoint = ModelCheckpoint(modelFile, monitor='val_loss', verbose=1, save_best_only=True)

model = model.model()
model.fit_generator(
    trainDataGen,
    steps_per_epoch=BATCH_SIZE,
    validation_data=valDataGen,
    validation_steps=BATCH_SIZE,
    callbacks=[earlyStopping, lrScheduler, modelCheckpoint],
    verbose=1,
    epochs=EPOCHS
)

