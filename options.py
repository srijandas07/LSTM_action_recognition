import argparse

def parse():
    print('Parsing arguments')
    parser = argparse.ArgumentParser(description='LSTM for AR')
   
    # All the options are by default set for toyota smarthome dataset
    parser.add_argument('--dataset', default='Smarthome') 
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--data_dim', default=39, type=int) #LCRNet output
    parser.add_argument('--num_classes', default=31, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_neuron', default=512, type=int)
    parser.add_argument('--n_dropout', default=0.5, type=float)
    parser.add_argument('--timesteps', default=30, type=int)
    parser.add_argument('--name', default='test')
    parser.add_argument('--epochs', default=100, type=int) 

    args = parser.parse_args()
    return args
