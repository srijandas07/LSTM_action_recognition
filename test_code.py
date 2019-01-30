import numpy as np
import keras
import pandas as pd
import os       


batch_size = 1000
dim = 39
step = 30
n_classes = 35

list_IDs = [i.strip() for i in open('/data/stars/user/sdas/smarthomes_data/splits/train_CS.txt').readlines()]
list_IDs_temp = [list_IDs[k] for k in range(1000, 2000)]
path = '/data/stars/user/rdai/smarthomes/smarthome_clipped_npz/'
X = np.empty((batch_size, step, dim))
f = 'Cook_p15_r03_v16_c03.mp4'
for i, ID in enumerate(list_IDs_temp):
    # Store sample 
    unpadded_file = np.load(path + os.path.splitext(ID)[0] + '.npz')['arr_0']
    if len(unpadded_file)>0:   
       f = ID 
    if len(unpadded_file)==0:
        unpadded_file = np.load(path + os.path.splitext(f)[0] + '.npz')['arr_0']
        list_IDs_temp[i] = f
    origin = unpadded_file[0, 3:6]   #Extract hip of the first frame
    [row, col] = unpadded_file.shape
    origin = np.tile(origin, (row, 13)) #making equal dimension
    unpadded_file = unpadded_file - origin  #translation
    extra_frames = (len(unpadded_file) % step)
    l = 0
    if len(unpadded_file)<step:
        extra_frames = step - len(unpadded_file)
        l = 1
    if extra_frames < (step/2) & l==0:
       padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
    else:
       [row, col] = unpadded_file.shape
       alpha = int(len(unpadded_file)/step) + 1
       req_pad = np.zeros(((alpha * step)-row, col))
       padded_file = np.vstack((unpadded_file, req_pad))
    splitted_file = np.split(padded_file, step)
    splitted_file = np.asarray(splitted_file)
    row, col, width = splitted_file.shape
    sampled_file = []
    for k in range(0,step):
        c = np.random.choice(col,1)
        sampled_file.append(splitted_file[k,c,:])
    sampled_file = np.asarray(sampled_file)
    X[i,] = np.squeeze(sampled_file)