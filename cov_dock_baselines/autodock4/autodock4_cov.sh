
# autodock vina docker
docker run -it -v $PWD:/data --rm ghcr.io/metaphorme/vina-all:v1.2.5

export AdCovScriptRoot=/data/autodock4/adCovalentDockResidue/adcovalent
export MGLROOT=/data/autodock4/mgltools_x86_64Linux2_1.5.7
export AutoDockRoot=/data/autodock4/x86_64Linux2



# 1. download covalent docking scripts from https://autodock.scripps.edu/resources/covalent-docking/
# done manually

# 2. download autodock 4.2.6
wget https://autodock.scripps.edu/wp-content/uploads/sites/56/2021/10/autodocksuite-4.2.6-x86_64Linux2.tar
tar -xvf autodocksuite-4.2.6-x86_64Linux2.tar

# 3. download MGLTools
# download from https://ccsb.scripps.edu/mgltools/download/491/
# done manually
cd mgltools_x86_64Linux2_1.5.7
bash install.sh
source initMGLtools.sh

export MGLROOT=$(pwd)
export LD_LIBRARY_PATH=$MGLROOT/lib/:$LD_LIBRARY_PATH
# export BABEL_LIBDIR="$MGL_ROOT/lib/openbabel/2.4.1"
# export BABEL_DATADIR="$MGL_ROOT/share/openbabel/2.4.1"



# step 1)
# input ligand is process by openbabel, so .mol2 or .sdf both ok
pythonsh ./adcovalent/prepareCovalent.py --ligand ./3upo_test/ligand.mol2 \
                --ligindices 1,2\
                --receptor ./3upo_test/3upo_protein.pdb\
                --residue B:SER222\
                --outputfile ./3upo_test/ligcovalent.pdb

# or
# pythonsh ./adcovalent/prepareCovalent.py --ligand ./3upo_test/ligand.mol2 \
#                 --ligindices 4,3\
#                 --ligsmart "C(=O)-O-C"\
#                 --receptor ./3upo_test/3upo_protein.pdb\
#                 --residue B:SER222\
#                 --outputfile ./3upo_test/ligcovalent.pdb

# step 2)
# [ this generates the file "3upo_protein.pdbqt" ]
pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r ./3upo_test/3upo_protein.pdb -A hydrogens

# and the covalent ligand that has been aligned:
# [ this generates the file "ligcovalent.pdbqt" ]
pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r ligcovalent.pdb 

# step 3)
# [ this generates the files "3upo_protein_rigid.pdbqt" and  "3upo_protein_flex.pdbqt" ]
pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_flexreceptor4.py -r 3upo_protein.pdbqt -s 3upo_protein:B:SER222


#    [ this generates the files "ligcovalent_rigid.pdbqt" and  "ligcovalent_flex.pdbqt" ]
pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_flexreceptor4.py -r ligcovalent.pdbqt -s ligcovalent:B:SER222

# step 4)
# The GPF for AutoGrid is generated with:

pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_gpf4.py -r 3upo_protein_rigid.pdbqt\
            -x ligcovalent_flex.pdbqt\
            -l ligcovalent_flex.pdbqt\
            -y -I 20\
            -o 3upo_protein.gpf

# The DPF for AutoDock is generated with:

pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_dpf4.py -r 3upo_protein_rigid.pdbqt\
            -x ligcovalent_flex.pdbqt\
            -l ligcovalent_flex.pdbqt\
            -o ligcovalent_3upo_protein.dpf\
            -p move='empty'

# change the config in .dpf file from 
#     unbound_model bound
# to 
#     unbound_energy 0.0


# step5) run autogrid and autodock

autogrid4 -p 3upo_protein.gpf -l 3upo_priotein.glg
../../x86_64Linux2/autodock4 -p ligcovalent_3upo_protein.dpf -l  ligcovalent_3upo_protein.dlg

# convert the output dlg file to sdf
# more convert option ref to https://openbabel.org/docs/FileFormats/AutoDock_PDBQT_format.html (add them after -a(read) -x(write))
# obabel -ipdbqt ligcovalent_3upo_protein.dlg -osdf -O result.sdf -ad -xn # do not use this, sdf file may be wrong by openbabel output
obabel -ipdbqt ligcovalent_3upo_protein.dlg -opdb -O result.pdb -ad


