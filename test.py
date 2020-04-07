from keras.models import load_model
from data import Data
import numpy as np
import sys
import keras

testData = sys.argv[1]
modelFile = sys.argv[2]

BATCH_SIZE = 1 
NUM_CLASSES = 10

dataObj = Data(BATCH_SIZE) 
testDataGen = dataObj.loadTestData(testData)

model = load_model(modelFile) 
loss, acc = model.evaluate_generator(testDataGen)

print("Accuracy- ", acc * 100) 
print("Error- ", (1-acc) * 100)

