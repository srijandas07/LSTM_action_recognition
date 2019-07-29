import numpy as np
import keras
import pandas as pd

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path_video_files, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.path = '/data/stars/user/sdas/MSRDailyActivity3D/LSTM_poses/'
        self.list_IDs = [i.strip() for i in open(path_video_files).readlines()]       
        self.n_classes = 16
        self.step = 10
        self.dim = 60
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.list_IDs)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.step, self.dim))
        y = np.empty((self.batch_size), dtype=int)
  
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            unpadded_file = np.load(self.path + ID + '.npz')['arr_0']
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 20))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X[i,] = np.squeeze(sampled_file)

            # Store class
            #y[i] = self.labels[ID]
        y = np.array([int(i[1:3]) for i in list_IDs_temp]) - 1      

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
