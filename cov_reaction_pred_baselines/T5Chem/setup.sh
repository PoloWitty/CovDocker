# desc:	env setup process for t5chem
# author:	Yangzhe Peng
# date:	2024/05/12



conda create -n t5chem python==3.8
conda install mkl==2023.0
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install transformers==4.10.2
conda install scikit-learn==0.24.1
conda install scipy==1.6.0
pip install rdkit==2023.9.5
conda install tensorboard
conda install pandas
conda install -c pytorch torchtext

# sample data: https://github.com/HelloJocelynLu/t5chem/blob/main/data/sample_data.tar.bz2
tar -xjvf sample_data.tar.bz2

# pretrained weight
wget https://yzhang.hpc.nyu.edu/T5Chem/models/USPTO_MT_model.tar.bz2

git clone https://github.com/HelloJocelynLu/t5chem.git
cd t5chem
git checkout develop
cd ..
python t5chem/t5chem/__main__.py -v

# python t5chem/t5chem/__main__.py train --data_dir data/sample/product/ --output_dir save/ --task_type product --pretrain ./models/USPTO_500_MT --num_epoch 30
# python t5chem/t5chem/__main__.py predict --data_dir data/sample/product/ --model_dir model/