module load cuda/8.0 cudnn/5.1-cuda-8.0
export PYTHONPATH=./model_scripts/:$PYTHONPATH
export PYTHONPATH=./readers/:$PYTHONPATH
mkdir -p weights_$2
python $1 $2 --epochs $3 --mode $4
