#!/bin/bash

# 1. Change to the home directory
cd ~

# 2. Clone the GitHub repository
git clone https://github.com/huynhnhathao/deeplearningmodels.git

# 3. Change to the cloned repository directory
cd deeplearningmodels

# 4. Create a virtual environment named 'env'
python3 -m venv env

# 5. Activate the virtual environment
source env/bin/activate

# 6. Install the requirements from requirements.txt
pip install -r requirements.txt

mlflow server --host 0.0.0.0 --port 8888

python train_bert_with_mix_precision.py --learning_rate 3e-5 --num_epoch 5 --l2_regularization 0.01 