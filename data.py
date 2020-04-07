from keras.preprocessing.image import *

class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def loadTrainData(self, trainData):
        datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255, featurewise_center=True)
        trainGenerator = datagen.flow_from_directory(trainData, subset='training', batch_size=self.batch_size, target_size=(256, 256))
        valGenerator = datagen.flow_from_directory(trainData, subset='validation', batch_size=self.batch_size, target_size=(256, 256))

        return trainGenerator, valGenerator

    def loadTestData(self, testData):
        datagen = ImageDataGenerator(rescale=1./255, featurewise_center=True)
        testGenerator = datagen.flow_from_directory(testData, target_size=(256, 256))

        return testGenerator

