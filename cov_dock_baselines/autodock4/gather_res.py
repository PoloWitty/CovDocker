import pandas as pd
import re
import os
import numpy as np
from typing import Optional
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel
from tabulate import tabulate
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import subprocess
import argparse
import json


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

def _find_pocket_bond_idx(pdb_filename, pocket_bond_list, ignore_H=True):
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
    ter_cnt = 0
    H_cnt = 0
    D_cnt = 0
    for i, line in enumerate(lines):
        if line.startswith('TER'):
            ter_cnt += 1
            continue
        name = line[12:16].strip(); resName = line[17:20].strip()
        chainID = line[21]; resSeq = line[22:26].strip()
        if ignore_H and name.startswith('H'):
            H_cnt += 1
            continue
        if name.startswith('D'):
            D_cnt += 1
            continue
        if [name,resName,chainID,resSeq] == pocket_bond_list:
            return i - ter_cnt - H_cnt - D_cnt
    return None

def get_pdb_mol(ligand_pdb_string, template_mol, bond_info):
    # bond_ligand_atom, bond_ligand_het, bond_ligand_chain, bond_ligand_chain_idx = bond_info
    pdb_complex = AllChem.MolFromPDBBlock(ligand_pdb_string)
    new_mol, sdf_string = None, None
    for pdb_mol in Chem.GetMolFrags(pdb_complex, asMols=True):
        pdb_info = pdb_mol.GetAtomWithIdx(0).GetPDBResidueInfo()
        pdb_ligand_het, pdb_ligand_chain, pdb_ligand_chain_idx = pdb_info.GetResidueName(), pdb_info.GetChainId(), str(pdb_info.GetResidueNumber())
        if bond_info[1:] != [pdb_ligand_het, pdb_ligand_chain, pdb_ligand_chain_idx]:
            continue
        try:
            new_mol = AllChem.AssignBondOrdersFromTemplate(template_mol, pdb_mol)
        except:
            continue
        if new_mol.GetNumAtoms() == template_mol.GetNumAtoms(): # already found the exact match
            break
    return new_mol

def rmsd_func(holo_coords: np.ndarray, predict_coords: np.ndarray, mol: Optional[Chem.Mol] = None) -> float:
    """ Symmetric RMSD for molecules. """
    if predict_coords is not np.nan:
        sz = holo_coords.shape
        if mol is not None:
            # get stereochem-unaware permutations: (P, N)
            base_perms = np.array(mol.GetSubstructMatches(mol, uniquify=False))
            # filter for valid stereochem only
            chem_order = np.array(list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)))
            perms_mask = (chem_order[base_perms] == chem_order[None]).sum(-1) == mol.GetNumAtoms()
            base_perms = base_perms[perms_mask]
            noh_mask = np.array([a.GetAtomicNum() != 1 for a in mol.GetAtoms()])
            # (N, 3), (N, 3) -> (P, N, 3), ((), N, 3) -> (P,) -> min((P,))
            best_rmsd = np.inf
            for perm in base_perms:
                rmsd = np.sqrt(np.sum((predict_coords[perm[noh_mask]] - holo_coords) ** 2) / sz[-2])
                if rmsd < best_rmsd:
                    best_rmsd = rmsd

            rmsd = best_rmsd
        else:
            rmsd = np.sqrt(np.sum((predict_coords - holo_coords) ** 2) / sz[-2])
        return rmsd
    return 1000.0

def calc_inter_bond_dist_error(predict_coords, pocket_coords, holo_coords, inter_bond):
    inter_bond_dist = np.linalg.norm(predict_coords[inter_bond[1]] - pocket_coords[inter_bond[0]], 2)
    inter_bond_dist_target = np.linalg.norm(holo_coords[inter_bond[1]] - pocket_coords[inter_bond[0]], 2)
    return np.sqrt((inter_bond_dist_target - inter_bond_dist)**2)

def calculate_rmsd(P, Q):
    """Calculate the RMSD between two aligned point clouds."""
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))

def optimal_assignment_rmsd(P, Q):
    """Find the optimal point correspondence between P and Q to minimize RMSD."""
    # Calculate the distance matrix between all pairs of points
    distance_matrix = cdist(P, Q, metric='euclidean')
    
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # Reorder Q according to the optimal assignment
    Q_reordered = Q[col_ind]
    
    # Calculate RMSD
    rmsd = calculate_rmsd(P, Q_reordered)
    return rmsd

# # Example usage:
# P = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])  # Point cloud 1
# Q = np.array([[4, 5, 6], [7, 8, 9], [1, 2, 3]]) # Point cloud 2 (shuffled)

# rmsd = optimal_assignment_rmsd(P, Q)
# print(f"Minimum RMSD: {rmsd}")

def clean_sym(sym):
    sym = sym.strip().title()
    sym = sym.replace('+','')
    sym = sym.replace('-','')
    sym = ''.join(filter(lambda x: not x.isdigit(), sym)) # drop all number
    return sym

def get_info_from_pdb_file(pdb_filename, sym2idx):
    coords = []; syms = []
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('HETATM') and (not line.endswith('H')):
                atom_name = line[12:16].strip()
                if not atom_name.endswith('X'): # only consider lingand atoms
                    continue
                x = line[30:38]; y=line[38:46]; z=line[46:54]; sym=line[76:78]
                sym = clean_sym(sym)
                coord = [float(x), float(y), float(z)]
                coords.append(coord)
                syms.append(sym2idx[sym]) # convert to index
    return np.array(syms), np.array(coords)
            

def run(cmd):
    return subprocess.getoutput(cmd)

def reorder_rmsd(pred_filename, target_filename):
    assert pred_filename.endswith('.pdb') and target_filename.endswith('.pdb'), 'Only support pdb file (or maybe .xyz will also be ok in the future)'
    try:
        import rmsd
    except ImportError:
        raise ImportError('Please install the rmsd package by running `pip install rmsd`')
    from rmsd import NAMES_ELEMENT, reorder_hungarian, reorder_distance, reorder_inertia_hungarian
    
    P_atoms, P_coords = get_info_from_pdb_file(target_filename, NAMES_ELEMENT)
    Q_atoms, Q_coords = get_info_from_pdb_file(pred_filename, NAMES_ELEMENT)
    
    # reorder the atoms and coordinates
    Q_reordered = reorder_hungarian(P_atoms, Q_atoms, P_coords, Q_coords) # Align the principal intertia axis and then re-orders the input atom list and xyz coordinates using the Hungarian method (using optimized column results)
    Q_atoms = Q_atoms[Q_reordered]
    Q_coords = Q_coords[Q_reordered]
    reordered_rmsd = calculate_rmsd(P_coords, Q_coords)
    return reordered_rmsd
    


def print_results(rmsd_results):
    table = {
        "%RMSD<1.0": np.mean(rmsd_results < 1.0),
        "%RMSD<1.5": np.mean(rmsd_results < 1.5),
        "%RMSD<2.0": np.mean(rmsd_results < 2.0),
        "%RMSD<3.0": np.mean(rmsd_results < 3.0),
        "%RMSD<4.0": np.mean(rmsd_results < 4.0),
        "%RMSD<5.0": np.mean(rmsd_results < 5.0),
        # "avg RMSD": np.mean(rmsd_results),
        
        "failed num": np.sum(rmsd_results == float('inf'))
    }

    # use tabulate to show the result
    table_ = {k:[v] for k,v in table.items()}
    print(tabulate(table_, headers="keys"))
    return table

def extract_time_from_dlg_file(dock_log_filename:str):
    time_block = None
    with open(dock_log_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Real='):
                # Real= 5m 37.78s,  CPU= 5m 37.56s,  System= 0.14s
                time_block = line.strip()
    assert time_block != None

    # Step 1: Extract the Real time part
    real_time_part = re.search(r"Real= (\d+m \d+\.\d+s)", time_block)
    if real_time_part:
        real_time = real_time_part.group(1)
        # Step 2: Extract minutes and seconds
        minutes, seconds, _ = re.findall(r"(\d+)", real_time)
        minutes = int(minutes)
        seconds = float(seconds)
    else:
        real_time_part = re.search(r"Real= (\d+\.\d+s)", time_block)
        real_time = real_time_part.group(1)
        seconds, _ = re.findall(r"(\d+)", real_time)
        seconds = float(seconds)
        minutes = 0


    # Convert to total seconds (optional)
    total_seconds = minutes * 60 + seconds
    return total_seconds

def run(cmd):
    subprocess.run(cmd, shell=True, check=True, timeout=60*15)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default='./res/', help='path to store vina output')
    parser.add_argument("--run-id", type=str, default='default-run-id')
    parser.add_argument("--use-wandb",type=int, default=0, help='whether use wandb to store result')
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()
    
    
    exhaustiveness=8
    if args.only_unseen:
        df = pd.read_csv(args.work_dir+'/dataset.unseen.csv')
        df = df[df['set']=='unseen']
    else:
        df = pd.read_csv(args.work_dir+'/dataset.csv')
        df = df[df['set']=='test']
    
    failed_pdb = []
    rmsd_list = []
    total_sec_list = []
    for i,row in tqdm.tqdm(df.iterrows(),desc='calc metric', total=len(df)):
        pdb_id = row['pdb_id']
        flex_res_filename = args.work_dir + f'./bonded/{pdb_id}/result.pdb'
        dock_log_filename = args.work_dir + f'./bonded/{pdb_id}/result.dlg'
        rigid_res_filename = args.work_dir + f'./bonded/{pdb_id}/ligcovalent_rigid.pdbqt'
        tar_flex_filename = args.work_dir + f'./bonded/{pdb_id}/ligcovalent_gt_flex.pdb'
        tar_filename = args.work_dir + f'./bonded/{pdb_id}/ligcovalent_gt.pdb'
        res_filename = args.work_dir + f'./bonded/{pdb_id}/total_res.pdb'

        if (not os.path.exists(flex_res_filename)) or (not os.path.exists(tar_filename)):
            failed_pdb.append(pdb_id)
            rmsd_list.append(float('inf'))
        else:
            # convert rigid res pdbqt to pdb
            convert_cmd = f'''
            $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/pdbqt_to_pdb.py -f {rigid_res_filename} -o {rigid_res_filename.replace('.pdbqt','.pdb')}
            '''
            try:
                run(convert_cmd)
            except:
                failed_pdb.append(pdb_id)
                rmsd_list.append(float('inf'))
                continue
            
            # concat rigid and flex part for res
            cat_cmd = f'''
            cat {rigid_res_filename.replace('.pdbqt','.pdb')} {flex_res_filename} > {res_filename}
            '''
            try:
                run(cat_cmd)
            except:
                failed_pdb.append(pdb_id)
                rmsd_list.append(float('inf'))
                continue

        

            # ATOM      1  CA  CYS B 491     -37.354  23.950  -7.796  1.00  0.00           C      
            
            # res_mol = Chem.MolFromPDBFile(flex_res_filename)
            # tar_mol = Chem.MolFromPDBFile(tar_filename)
            # obmol, sdf_string, atom_types, atom_coords, bond_table = parse_mol(flex_res_filename, filetype='pdb', string=False, remove_H=True, generate_conformer=False)
            # parse_mol(tar_filename, filetype='pdb', string=False, remove_H=True, generate_conformer=False)
                 
            # pred_coords = []
            # with open(res_filename, 'r') as f:
            #     lines = f.readlines()
            #     for line in lines:
            #         line = line.strip()
            #         if line.startswith('HETATM') and (not line.endswith('H')):
            #             x = line[32:38]; y=line[38:46]; z=line[46:54]
            #             coord = [float(x), float(y), float(z)]
            #             # line = line.strip().split()
            #             # try:
            #             #     coord = list(map(float, line[-7:-4]))
            #             # except:
            #             pred_coords.append(coord)
            
            
            # tar_coords = []
            # with open(tar_filename, 'r') as f:
            #     lines = f.readlines()
            #     for line in lines:
            #         line = line.strip()
            #         if line.startswith('HETATM') and (not line.endswith('H')):
            #             x = line[32:38]; y=line[38:46]; z=line[46:54]
            #             coord = [float(x), float(y), float(z)]
            #             # line = line.strip().split()
            #             # coord = list(map(float, line[-7:-4]))
            #             tar_coords.append(coord)
            # pred_coords = np.array(pred_coords)
            # tar_coords = np.array(tar_coords)

            
            try:
                # _rmsd = optimal_assignment_rmsd(pred_coords, tar_coords)
                _rmsd = reorder_rmsd(res_filename, tar_filename)
            except:
                failed_pdb.append(pdb_id)
                rmsd_list.append(float('inf'))
                continue

            total_sec = extract_time_from_dlg_file(dock_log_filename)
            total_sec_list.append(total_sec)

            # calc rmsd
            print(f'pdb_id:{pdb_id}-RMSD:{_rmsd}-time:{total_sec}')
            rmsd_list.append(_rmsd)
    
    print(f'failed pdb: {len(failed_pdb)}')
    print(failed_pdb)
# failed pdb: 71
# ['7A2A', '7AWE', '6Y6U', '7A47', '6YEO', '6WTT', '7NXK', '7NXW', '7BDT', '7DUQ', '7JKV', '5RER', '6XY5', '7ONB', '7CC2', '5RES', '6ZZ1', '6ZBW', '7E9A', '7KRF', '6YPY', '6WZV', '7KCW', '7BZL', '7NWS', '7BFW', '7FD5', '7O6K', '7BA6', '7NVI', '7BAB', '7A1W', '7KGM', '6YPD', '7ATX', '6Y1N', '7A72', '7DUR', '7C7K', '6Y49', '7EVM', '7KIV', '7AU1', '5RFU', '6X1M', '7K8K', '7FD4', '7NJ9', '7KHQ', '7CMM', '7BG3', '7RFU', '7RN1', '6YCC', '6X6C', '7AZ1', '6M60', '7KRC', '7B4H', '7O3P', '7D3I', '7BIW', '7DNC', '7NXT', '6ZBX', '5RFH', '5RH5', '6YG2', '6YEN', '5RFP', '5RFR']
    rmsd = np.array(rmsd_list)
    reorder_res_table = print_results(rmsd)
    print(f'avg time: {sum(total_sec_list)/len(total_sec_list)}')
    
    if args.use_wandb:
        import wandb
        wandb.init(
            project="res_dock",
            config = args
        )
        config_dict = {
            'Average dock time cost': sum(total_sec_list)/len(total_sec_list),
            'Total dock time cost': sum(total_sec_list),
            'failed num': len(failed_pdb),
            'exhaustiveness': exhaustiveness
        }
        wandb.config.update(config_dict)
        log_dict = {
            'res': reorder_res_table,
            'reorder_res': reorder_res_table,
        }
        wandb.log(log_dict)
