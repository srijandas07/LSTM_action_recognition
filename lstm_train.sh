export PATH=/home/sdas/anaconda2/bin:$PATH
module load cuda/8.0 cudnn/5.1-cuda-8.0
export PYTHONPATH=./model_scripts/:$PYTHONPATH
export PYTHONPATH=./readers/:$PYTHONPATH
mkdir -p weights_$2
python ./lstm_train_skeleton.py $1 $2
