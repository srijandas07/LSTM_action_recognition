from argparse import ArgumentParser
import sys

argv = sys.argv
# Parse arguments                                                                                                                                                                        
parser = ArgumentParser(description='Helps getting things done.')
# parser.add_argument('-v', '--verbose', action='store_true', help='run verbose')                                                                                                        
parser.add_argument('name', help='Experiment Name')
parser.add_argument('-epochs', '--epochs', help='Number of Epochs', default=30)
args = parser.parse_args(argv[1:])

from ntu_skeleton_loader_sampling import DataGenerator
timesteps = 30
   
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.layers import TimeDistributed, GaussianNoise, GaussianDropout, Dropout
from keras.models import Model
import numpy as np
import scipy.io 
import keras
import h5py
import itertools
#import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from models import build_model_without_TS                                                       
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback


class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):

        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')

data_dim = 39
num_classes = 35
batch_size = 200
n_neuron = 512
n_dropout = 0.5
name = args.name

csvlogger = CSVLogger(name+'_smarthomes.csv')
epochs = int(args.epochs)
model = build_model_without_TS(n_neuron, n_dropout, batch_size, timesteps, data_dim, num_classes)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.005, clipnorm=1), metrics=['accuracy']) 

train_generator = DataGenerator('/data/stars/user/sdas/smarthomes_data/splits/train_CS.txt', batch_size = batch_size)
val_generator = DataGenerator('/data/stars/user/sdas/smarthomes_data/splits/validation_CS.txt', batch_size = batch_size)
test_generator = DataGenerator('/data/stars/user/sdas/smarthomes_data/splits/test_CS.txt', batch_size = batch_size)

model_checkpoint = CustomModelCheckpoint(model, './weights_'+name+'/epoch_')

model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=False,
                    epochs=epochs,
                    callbacks = [csvlogger, model_checkpoint],
                    workers=6)

model_json = model.to_json()
with open("./models/model_"+name+".json", "w") as json_file:
	json_file.write(model_json)
print("Saved model to disk")
print(model.evaluate_generator(generator = test_generator))
