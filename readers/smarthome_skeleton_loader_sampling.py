import numpy as np
import keras
import pandas as pd
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path_video_files, batch_size=32):
        'Initialization'
        self.batch_size = batch_size
        self.path = '/data/stars/user/rdai/smarthomes/smarthome_clipped_npz/'
        self.list_IDs = [i.strip() for i in open(path_video_files).readlines()]       
        self.n_classes = 35
        self.step = 30
        self.dim = 39
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

    def _name_to_int(self, name):
        integer=0
        if name=="Cook":
            integer=1
        elif name=="Cook.Cleandishes":
            integer=2
        elif name=="Cook.Cleanup":
            integer=3
        elif name=="Cook.Cut":
            integer=4
        elif name=="Cook.Stir":
            integer=5
        elif name=="Cook.Usestove":
            integer=6
        elif name=="Cutbread":
            integer=7
        elif name=="Drink":
            integer=8
        elif name=="Drink.Frombottle":
            integer=9
        elif name=="Drink.Fromcan":
            integer=10
        elif name=="Drink.Fromcup":
            integer=11
        elif name=="Drink.Fromglass":
            integer=12
        elif name=="Eat.Attable":
            integer=13
        elif name=="Eat.Snack":
            integer=14
        elif name=="Enter":
            integer=15
        elif name=="Getup":
            integer=16
        elif name=="Laydown":
            integer=17
        elif name=="Leave":
            integer=18
        elif name=="Makecoffee":
            integer=19
        elif name=="Makecoffee.Pourgrains":
            integer=20
        elif name=="Makecoffee.Pourwater":
            integer=21
        elif name=="Maketea.Boilwater":
            integer=22
        elif name=="Maketea.Insertteabag":
            integer=23
        elif name=="Pour.Frombottle":
            integer=24
        elif name=="Pour.Fromcan":
            integer=25
        elif name=="Pour.Fromcup":
            integer=26
        elif name=="Pour.Fromkettle":
            integer=27
        elif name=="Readbook":
            integer=28
        elif name=="Sitdown":
            integer=29
        elif name=="Takepills":
            integer=30
        elif name=="Uselaptop":
            integer=31
        elif name=="Usetablet":
            integer=32
        elif name=="Usetelephone":
            integer=33
        elif name=="Walk":
            integer=34
        elif name=="WatchTV":
            integer=35
        return integer

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.step, self.dim))
        y = np.empty((self.batch_size), dtype=int)
  
        # Generate data
        f = 'Cook_p15_r03_v16_c03.mp4'
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            unpadded_file = np.load(self.path + os.path.splitext(ID)[0] + '.npz')['arr_0']
 
            if len(unpadded_file)>0:   
               f = ID 
            if len(unpadded_file)==0:
                unpadded_file = np.load(self.path + os.path.splitext(f)[0] + '.npz')['arr_0']
                list_IDs_temp[i] = f
            origin = unpadded_file[0, 3:6]   #Extract hip of the first frame
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 13)) #making equal dimension
            unpadded_file = unpadded_file - origin  #translation
            extra_frames = (len(unpadded_file) % self.step)
            l = 0
            if len(unpadded_file)<self.step:
                extra_frames = self.step - len(unpadded_file)
                l = 1
            if extra_frames < (self.step/2) & l==0:
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = int(len(unpadded_file)/self.step) + 1
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

        #y = np.array([int(i[-3:]) for i in list_IDs_temp]) - 1      
        y = np.array([int(self._name_to_int(i.split('_')[0])) for i in list_IDs_temp]) - 1
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
