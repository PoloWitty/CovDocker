
# # (Optional) autodock vina docker
# docker run -it -v $PWD:/data ghcr.io/metaphorme/vina-all:v1.2.5

# export AdCovScriptRoot=/data/adCovalentDockResidue/adcovalent
# export AutoDockRoot=/data/x86_64Linux2
# export MGLROOT=/data/mgltools_x86_64Linux2_1.5.7
# export LD_LIBRARY_PATH=$MGLROOT/lib/:$LD_LIBRARY_PATH
# source $MGLROOT/initMGLtools.sh

# 1. download covalent docking scripts from https://autodock.scripps.edu/resources/covalent-docking/
wget https://autodock.scripps.edu/download/468/ 
mv index.html adCovalentDockResidue_v1.2.tar.gz
tar -xzvf adCovalentDockResidue_v1.2.tar.gz
# and uncomment prepareCovalent.py L781's return to escape -nan value for ground truth value ( found this by PYZ 2024/05/14)
# other files are the same as orig
# done manually
export AdCovScriptRoot=$(pwd)/adCovalentDockResidue/adcovalent

# 2. download autodock 4.2.6
wget https://autodock.scripps.edu/wp-content/uploads/sites/56/2021/10/autodocksuite-4.2.6-x86_64Linux2.tar
tar -xvf autodocksuite-4.2.6-x86_64Linux2.tar
export AutoDockRoot=$(pwd)/x86_64Linux2

# 3. download MGLTools
# download from https://ccsb.scripps.edu/mgltools/download/491/
wget https://ccsb.scripps.edu/mgltools/download/491/ 
mv index.html mgltools_x86_64Linux2_1.5.7p1.tar.gz
tar -xzvf mgltools_x86_64Linux2_1.5.7p1.tar.gz

# setup MGLTools
pushd mgltools_x86_64Linux2_1.5.7
bash install.sh
source initMGLtools.sh
popd
export MGLROOT=$(pwd)/mgltools_x86_64Linux2_1.5.7
export LD_LIBRARY_PATH=$MGLROOT/lib/:$LD_LIBRARY_PATH
# export BABEL_LIBDIR="$MGL_ROOT/lib/openbabel/2.4.1"
# export BABEL_DATADIR="$MGL_ROOT/share/openbabel/2.4.1"


# 4. install need python packages in vina env
pip install tqdm
pip install tabulate
pip install wandb
pip install rmsd

# if there is Error: libxml2.so.2: cannot open shared object file: No such file or directory
apt-get install libxml2