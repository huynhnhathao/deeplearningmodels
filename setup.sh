#!/bin/bash

python3 -m venv env

source env/bin/activate

pip install -r requirements.txt

nohup mlflow server --host 0.0.0.0 --port 8888 &

python train_bert_with_mix_precision.py --learning_rate 3e-5 --num_epoch 5 --l2_regularization 0.01 