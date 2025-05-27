
# no need to install vina by url
# wget https://vina.scripps.edu/wp-content/uploads/sites/55/2020/12/autodock_vina_1_1_2_linux_x86.tgz
# tar xzvf autodock_vina_1_1_2_linux_x86.tgz
# export VINA_PATH=$(pwd)

conda create -n vina python=3.8
conda activate vina
pip install vina

python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
git clone https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3.git

pip install rdkit==2023.9.5
pip install pandas
pip install tqdm
pip install openbabel-wheel
pip install tabulate
pip install rmsd
pip install wandb