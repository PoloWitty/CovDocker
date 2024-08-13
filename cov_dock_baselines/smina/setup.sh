conda create -n smina python=3.8
conda activate smina

wget https://nchc.dl.sourceforge.net/project/smina/smina.static?viasf=1
mv smina.static\?viasf\=1 smina
chmod +x smina
export SMINA_PATH=$(pwd)

pip install rdkit==2023.9.5
pip install openbabel-wheel
pip install pandas
pip install tqdm
pip install tabulate
pip install rmsd
pip install wandb