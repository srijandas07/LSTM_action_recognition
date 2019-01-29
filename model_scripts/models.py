from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.layers import TimeDistributed, GaussianNoise, GaussianDropout, Dropout
from keras.models import Model


def build_model_without_TS(n_neuron, n_dropout, batch_size, timesteps, data_dim, num_classes):
    print('Build model!!!') 
    model = Sequential()                       
    model.add(LSTM(n_neuron, return_sequences=True, input_shape=(timesteps, data_dim)))
    model.add(LSTM(n_neuron, return_sequences=True))
    model.add(LSTM(n_neuron))
    model.add(Dropout(n_dropout))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_model_with_TS(n_neuron, n_dropout, batch_size, timesteps, data_dim, num_classes):
    print('Build model...')         
    model = Sequential()
    model.add(LSTM(n_neuron, return_sequences=True, batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(LSTM(n_neuron, return_sequences=True))
    model.add(LSTM(n_neuron, return_sequences=True))
    model.add(Dropout(n_dropout))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    return model

