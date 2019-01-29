This script takes the 3D skeleton as input and trains a 3-layer LSTM.
Two models of LSTMs are defined in the model.py script (You can use any one of them).
For demo - the location of pre-processed 3D skeleton files are mentioned in the lstm_train.sh
script. You can change this location for processing it on your dataset.
For other dataset, you also need to change the dataloaders.

python lstm_train.py -h can display the parameters required by the python file to train the LSTM.

Example- 
sh lstm_train.sh test sampling 10
            OR
sh lstm_train.sh test translation 10

Thus, the input parameters to the python file is 1)name of the experiment
2)input data_location, 3)mode of data sampling required (translation, or translation+sampling)
and 4)number of epochs.

The script will generate a weight directory in the name of the experiment, where the models will be saved after every epoch.
It will also  generate a csv file with the training details. The best model should be used for testing using the evaluation_model.py
script.

For evaluating - 
python evaluating_model.py -h will specify the input parameters required to evaluate the model.
Example - 
python evaluating_model.py test --skeleton_location /data/stars/user/sdas/NTU_RGB/skeleton_csv/ --data_type translation --model_name epochs_2
 
implies -> python evaluate_model.py -name_of_the_experiment -input_skeleton_location -type_of_sampling -model_name

Enjoy with LSTM!!!
