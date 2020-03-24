from options import parse
import sys

from smarthome_skeleton_loader_sampling import DataGenerator
import numpy as np
import keras
import h5py
import pandas as pd
from models import build_model_without_TS                                                       
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from multiprocessing import cpu_count


class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):

        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')

if __name__ == '__main__':
	args = parse()
	csvlogger = CSVLogger(args.name+'_smarthomes.csv')
	model = build_model_without_TS(args.n_neuron, args.n_dropout, args.batch_size, args.timesteps, args.data_dim, args.num_classes)
	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=args.lr, clipnorm=1), metrics=['accuracy']) 

	#training, testing splits are in splits directory
	train_generator = DataGenerator('../splits/train_CS.txt', batch_size = batch_size)
	val_generator = DataGenerator('../splits/validation_CS.txt', batch_size = batch_size)
	test_generator = DataGenerator('../splits/test_CS.txt', batch_size = batch_size)

	if not os.path.exists('./weights_'+args.name):
		os.makedirs('./weights_'+args.name)
	model_checkpoint = CustomModelCheckpoint(model, './weights_'+args.name+'/epoch_')

	model.fit_generator(generator=train_generator,
			    validation_data=val_generator,
			    use_multiprocessing=True,
			    epochs=args.epochs,
			    callbacks = [csvlogger, model_checkpoint],
			    workers=cpu_count()-2)


	print(model.evaluate_generator(generator = test_generator))
