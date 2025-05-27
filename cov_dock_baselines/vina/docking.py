import pickle
from vina import Vina
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import numpy as np
import pandas as pd
import tqdm
import multiprocessing
from openbabel import openbabel
import time
import json
import argparse

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

def dock_by_vina(protein_file, ligand_file, pocket_center, prefix, pdb_id, box_len=20, seed=0):
    v = Vina(sf_name='vina', seed=seed)

    v.set_receptor(protein_file)

    try:
        v.set_ligand_from_file(ligand_file)
    except:
        return 'read_ligand_error'
    
    try:
        v.compute_vina_maps(center=pocket_center, box_size=[20, 20, 20])
        energy = v.score()
    except:
        try:
            v.compute_vina_maps(center=pocket_center, box_size=[box_len, box_len, box_len])
            energy = v.score()
        except:
            v.compute_vina_maps(center=pocket_center, box_size=[1.5*box_len, 1.5*box_len, 1.5*box_len])
            energy = v.score()
    
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])

    try:
        # Minimized locally the current pose
        energy_minimized = v.optimize()
    except:
        return 'optimize error'
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    v.write_pose(f'{prefix}/{pdb_id}_ligand_minimized.pdbqt', overwrite=True)

    # Dock the ligand
    v.dock(exhaustiveness=exhaustiveness, n_poses=20)
    v.write_poses(f'{prefix}/{pdb_id}_ligand_vina_out_5.pdbqt', n_poses=5, overwrite=True)

    return 'success'


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


def parse(x):
    i,row = x
    pdb_id = row['pdb_id']
    
    # if pdb_id not in ['7KIA', '7JN7', '6XJ3', '6YEO', '6YCF', '6XR3', '6XRO', '7KIW', '7AU0', '6W2Z', '6VT1', '7RFR', '5RFM', '7KEP', '6XB2', '6VJE', '7E9A', '7KRF', '6WYG', '6XJP', '6Z4B', '6WZV', '7RC0', '7L9X', '7LLI', '6Y1L', '7LDL', '7ONK', '7A1W', '6VT8', '6YPD', '7KIS', '7LTN', '7ATX', '6Y1N', '7A72', '6Y49', '7AU1', '7AU8', '7NT1', '6WVQ', '6X1M', '7K8K', '6VH4', '7B2V', '7O6M', '6XJS', '6XJR', '7LGS', '6YCC', '7DTZ', '6X6C', '7R7H', '6XYC', '7RGL', '6WVO', '6WVC', '7LNR', '6LNQ', '7DNC', '7LY1', '7NE2', '7NMH', '7NH4', '7ATW', '7KCY', '7BH5', '6VIM', '5RFN', '5RFG']:
    #     return True
    
    # prefix = os.path.abspath(os.path.dirname(__file__))+f"/data/processed/bonded/{pdb_id}"
    prefix = args.work_dir+f"/bonded/{pdb_id}"
    origin_protein_file = f'{prefix}/{pdb_id}_10Apocket.pdb'
    
    _,_, pocket_atoms, pocket_coords, _ = parse_mol(origin_protein_file)
    pocket_center = np.array(pocket_coords).mean(axis=0)
    
    origin_ligand_file = f'{prefix}/{pdb_id}_ligand.sdf'

    mol = Chem.SDMolSupplier(origin_ligand_file)[0]
    # 获取构象
    conf = mol.GetConformer()

    # 计算每个原子与pocket_center之间的距离
    distances = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_coord = np.array([pos.x, pos.y, pos.z])
        distance = np.linalg.norm(atom_coord - pocket_center)
        distances.append(distance)
    # print(max(distances))
    # print(min(distances))

    smiles = Chem.MolToSmiles(mol)
    
    box_len = 2 * max(distances)
    print(box_len)

    protein_pdbqt_file = f"{prefix}/{pdb_id}_protein.pdbqt"
    ligand_sdf_rc_file = f"{prefix}/{pdb_id}_ligand_rc.sdf"
    ligand_mol2_rc_file = f"{prefix}/{pdb_id}_ligand_rc.mol2"
    ligand_pdbqt_rc_file = f"{prefix}/{pdb_id}_ligand_rc.pdbqt"
    ligand_docked_sdf_file = f"{prefix}/{pdb_id}_ligand_vina_out_5.sdf"

    generate_and_write_sdf_from_smiles_using_rdkit_E3Bind(smiles, rdkitMolFile=ligand_sdf_rc_file, shift_dis=pocket_center.tolist(), fast_generation=False)

    os.system(f'obabel -isdf {ligand_sdf_rc_file} -omol2 -O {ligand_mol2_rc_file}')
    os.system(f'python ./AutoDockTools_py3/AutoDockTools/Utilities24/prepare_receptor4.py -r {origin_protein_file} -o {protein_pdbqt_file}')
    os.system(f'python ./AutoDockTools_py3/AutoDockTools/Utilities24/prepare_ligand4.py -l {ligand_mol2_rc_file} -o {ligand_pdbqt_rc_file}')

    start_time = time.time()
    ret_value = dock_by_vina(protein_pdbqt_file, ligand_pdbqt_rc_file, pocket_center.tolist(), prefix, pdb_id, box_len=box_len, seed=seed)
    end_time  = time.time()
    time_cost = end_time - start_time
    if ret_value == 'read_ligand_error':
        print(f'{pdb_id} read_ligand_error')
        return pdb_id,0
    elif ret_value == 'success':
        print(f'{pdb_id} done')
        os.system(f'obabel -ipdbqt {prefix}/{pdb_id}_ligand_vina_out_5.pdbqt -osdf -O {ligand_docked_sdf_file}')
        return True, time_cost
    elif ret_value == 'optimize error':
        print(f'{pdb_id} optimize error')
        return pdb_id,0
    else:
        raise ValueError('Unknown error')
        return pdb_id,0

    

  

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work-dir", type=str, default='./data', help='Path to data, note that the output will be in this dir too')
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()
    
    exhaustiveness=8
    seed=args.seed
    
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
    time_cost_list = []
    with tqdm.trange(len(df), desc='docking') as pbar:
        with multiprocessing.Pool(10) as pool:
            for inner_output, time_cost in pool.imap(parse, df.iterrows()):
                if inner_output != True:
                    error_pdb.append(inner_output)
                else:
                    time_cost_list.append(time_cost)
                pbar.update()
    
    print(f"failed pdb {len(error_pdb)}")
    print(error_pdb)
    
    print(f"total run time: {np.sum(time_cost_list):.0f}s")
    print(f"average run time: {np.mean(time_cost_list):.2f}s")
    
    log_dict = {
        'Average dock time cost': np.mean(time_cost_list),
        'Total dock time cost': np.sum(time_cost_list),
        'failed num': len(error_pdb),
        'seed': args.seed,
        'exhaustiveness': exhaustiveness
    }
    
    json.dump(log_dict, open(args.work_dir+'/log.json', 'w'))
    
# failed pdb 13
# ['7AWE', '6YEO', '7CC2', '7E9A', '6WZV', '7FD5', '6YPD', '7ATX', '7AU1', '6X1M', '7FD4', '6X6C', '6YEN']
# total run time: 10372s
# average run time: 49.39s