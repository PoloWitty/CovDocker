import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
import pandas as pd
import tqdm
import multiprocessing
import subprocess
from openbabel import openbabel
import time
import argparse
import json

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500, confId=0)
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol

def write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()

def generate_and_write_sdf_from_smiles_using_rdkit_E3Bind(smiles, rdkitMolFile=None, shift_dis=[0.0, 0.0, 0.0], fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords - coords.mean(axis=0) + np.array(shift_dis)
    if rdkitMolFile is not None:
        write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)
    return new_coords

# modified from https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/main/rf2aa/data/parsers.py#L744
def parse_mol(filename, filetype="pdb", string=False, remove_H=True, generate_conformer: bool = False):
    """Parse small molecule file content.

    Parameters
    ----------
    filename : str
    filetype : str
    string : bool
        If True, `filename` is a string containing the molecule data.
    remove_H : bool
        Whether to remove hydrogen atoms.

    Returns
    -------
    obmol: OBMol
        openbabel molecule object representing the pocket
    sdf_string: str
        converted sdf file string
    atom_types: List[int] (N_atoms, )
    atom_coords: List[List[int]] (N_atoms, 3) float
        Atom coordinates
    bond_table: List[List[int]] (N_bonds, 2)
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(filetype)
    obmol = openbabel.OBMol()
    if string:
        obConversion.ReadString(obmol,filename)
    # elif filetype=='sdf':
    #     molstring = clean_sdffile(filename)
    #     obConversion.ReadString(obmol,molstring)
    else:
        obConversion.ReadFile(obmol,filename)
    if generate_conformer:
        builder = openbabel.OBBuilder()
        builder.Build(obmol)
        ff = openbabel.OBForceField.FindForceField("mmff94")
        did_setup = ff.Setup(obmol)
        if did_setup:
            ff.FastRotorSearch()
            ff.GetCoordinates(obmol)
        else:
            raise ValueError(f"Failed to generate 3D coordinates for molecule {filename}.")
    if remove_H:
        obmol.DeleteHydrogens()
        # the above sometimes fails to get all the hydrogens
        i = 1
        while i < obmol.NumAtoms()+1:
            if obmol.GetAtom(i).GetAtomicNum()==1:
                obmol.DeleteAtom(obmol.GetAtom(i))
            else:
                i += 1
    atom_types = [obmol.GetAtom(i).GetAtomicNum()
                 for i in range(1, obmol.NumAtoms()+1)]
    atom_coords = [[obmol.GetAtom(i).x(),obmol.GetAtom(i).y(), obmol.GetAtom(i).z()] 
                                for i in range(1, obmol.NumAtoms()+1)]# (natoms, 3)
    bond_table = [[b.GetBeginAtomIdx()-1, b.GetEndAtomIdx()-1, b.GetBondOrder()] for b in openbabel.OBMolBondIter(obmol)]
    obConversion.SetOutFormat("sdf")
    sdf_string = obConversion.WriteString(obmol)
    return obmol, sdf_string, atom_types, atom_coords, bond_table

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def parse(x):
    i,row = x
    
    pdb_id = row['pdb_id']
    
    # prefix = os.path.abspath(os.path.dirname(__file__))+f"/data/processed/bonded/{pdb_id}/"
    prefix = args.work_dir+f"/bonded/{pdb_id}"
    origin_protein_file = f'{prefix}/{pdb_id}_10Apocket.pdb'
    
    _,_, pocket_atoms, pocket_coords, _ = parse_mol(origin_protein_file)
    pocket_center = np.array(pocket_coords).mean(axis=0)

    origin_ligand_file = f'{prefix}/{pdb_id}_ligand.sdf'
    ligand_sdf_rc_file = f"{prefix}/{pdb_id}_ligand_rc.sdf"
    ligand_pdb_rc_file = f"{prefix}/{pdb_id}_ligand_rc.pdb"
    ligand_out_pdb_file = f"{prefix}/{pdb_id}_ligand_smina_exhaustiveness{exhaustiveness}.pdb"
    ligand_out_sdf_file = f"{prefix}/{pdb_id}_ligand_smina_exhaustiveness{exhaustiveness}_out.sdf"

    mol = Chem.SDMolSupplier(origin_ligand_file)[0]
    smiles = Chem.MolToSmiles(mol)
    generate_and_write_sdf_from_smiles_using_rdkit_E3Bind(smiles, rdkitMolFile=ligand_sdf_rc_file, shift_dis=pocket_center.tolist(), fast_generation=False)

    box_len = 20
    # remember to export $SMINA_PATH
    try:
        run(f'obabel -isdf {ligand_sdf_rc_file} -opdb -O {ligand_pdb_rc_file}')
        start_time = time.time()
        run(f'$SMINA_PATH/smina --ligand {ligand_pdb_rc_file} --receptor {origin_protein_file} \
                    --center_x {pocket_center.tolist()[0]} \
                    --center_y {pocket_center.tolist()[1]} \
                    --center_z {pocket_center.tolist()[2]} \
                    --size_x {box_len} \
                    --size_y {box_len} \
                    --size_z {box_len} \
                    --exhaustiveness {exhaustiveness} \
                    --out {ligand_out_pdb_file} \
                    --seed {args.seed}')
        end_time = time.time()
        cost_time = end_time - start_time
        # exhaustiveness=8 is also the default param in smina
        run(f'obabel -ipdb {ligand_out_pdb_file} -osdf -O {ligand_out_sdf_file}')
    except:
        return pdb_id,0

    return True, cost_time
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=str, default='./data', help='Path to data, note that the output will be in this dir too')
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()
    
    exhaustiveness=8
    
    if args.only_unseen:
        df = pd.read_csv(args.work_dir+'/dataset.unseen.csv')
        df = df[df['set']=='unseen']
    else:
        df = pd.read_csv(args.work_dir+'/dataset.csv')
        df = df[df['set']=='test']
    
    # for i,row in tqdm.tqdm(df.iterrows(), total=len(df)):
    #     parse((i,row))
    #     breakpoint()
    
    error_pdb = []
    cost_time_list = []
    with tqdm.trange(len(df), desc='prepare random conform and ligindices') as pbar:
        with multiprocessing.Pool(10) as pool:
            for inner_output,cost_time in pool.imap(parse, df.iterrows()):
                if inner_output != True:
                    error_pdb.append(inner_output)
                else:
                    cost_time_list.append(cost_time)
                pbar.update()
    
    print(f"failed pdb {len(error_pdb)}")
    print(error_pdb)
    # failed pdb: 0
    # total run time: 21min02s, 5.07s/it
    
    print(f"mean time cost {np.mean(cost_time_list)}s")
    print(f'total time cost {np.sum(cost_time_list)}s')
    #mean time cost 55.35893296233207s
    # total time cost 12345.042050600052s
    
    log_dict = {
        'Average dock time cost': np.mean(cost_time_list),
        'Total dock time cost': np.sum(cost_time_list),
        'failed num': len(error_pdb),
        'seed': args.seed,
        'exhaustiveness': exhaustiveness
    }
    
    json.dump(log_dict, open(args.work_dir+'/log.json', 'w'))