import pandas as pd
import os
import numpy as np
from typing import Optional
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel
from tabulate import tabulate
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

def get_info_from_pdb_file(pdb_filename, sym2idx):
    coords = []; syms = []
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('HETATM') and (not line.endswith('H')):
                x = line[30:38]; y=line[38:46]; z=line[46:54]; sym=line[76:78].strip().title()
                coord = [float(x), float(y), float(z)]
                coords.append(coord)
                syms.append(sym2idx[sym]) # convert to index
    return np.array(syms), np.array(coords)

def calculate_rmsd(P, Q):
    """Calculate the RMSD between two aligned point clouds."""
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))


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
    

def print_results(rmsd_results, inter_bond_dist_results):    
    table = {
        "%RMSD<1.0": np.mean(rmsd_results < 1.0),
        "%RMSD<1.5": np.mean(rmsd_results < 1.5),
        "%RMSD<2.0": np.mean(rmsd_results < 2.0),
        "%RMSD<3.0": np.mean(rmsd_results < 3.0),
        "%RMSD<4.0": np.mean(rmsd_results < 4.0),
        "%RMSD<5.0": np.mean(rmsd_results < 5.0),
        # "avg RMSD": np.mean(rmsd_results),
        
        "%Inter bond dist error < 0.5": np.mean(inter_bond_dist_results < 0.5),
        "%Inter bond dist error < 1": np.mean(inter_bond_dist_results < 1),
        "%Inter bond dist error < 2": np.mean(inter_bond_dist_results < 2),
        "%Inter bond dist error < 3": np.mean(inter_bond_dist_results < 3),
        # "avg Inter bond dist error": np.mean(inter_bond_dist_results),
        
        "failed num": np.sum(rmsd_results == float('inf'))
    }

    # use tabulate to show the result
    table_ = {k:[v] for k,v in table.items()}
    print(tabulate(table_, headers="keys"))
    return table
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default='./res/', help='path to store smina output')
    parser.add_argument("--run-id", type=str, default='default-run-id')
    parser.add_argument("--use-wandb",type=int, default=0, help='whether use wandb to store result')
    parser.add_argument("--dock-log-filename", type=str, default='')
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
    rmsd_list = []; reorder_rmsd_list = []
    inter_bond_dist_error_list = []
    for i,row in tqdm.tqdm(df.iterrows(),desc='calc metric', total=len(df)):
        pdb_id = row['pdb_id']
        res_filename = args.work_dir + f'./bonded/{pdb_id}/{pdb_id}_ligand_smina_exhaustiveness{exhaustiveness}_out.sdf'
        tar_filename = args.work_dir + f'./bonded/{pdb_id}/{pdb_id}_ligand.sdf'
        tar_pdb_filename = args.work_dir + f'./bonded/{pdb_id}/{pdb_id}_ligand.pdb'
        pocket_filename = args.work_dir + f'./bonded/{pdb_id}/{pdb_id}_10Apocket.pdb'
        reorder_tar_filename = args.work_dir + f'./bonded/{pdb_id}/{pdb_id}_ligand_reordered_tar.pdb'
        reorder_out_filename = args.work_dir + f'./bonded/{pdb_id}/{pdb_id}_ligand_reordered_out.pdb'
        
        if not os.path.exists(res_filename):
            failed_pdb.append(pdb_id)
            rmsd_list.append(float('inf')); reorder_rmsd_list.append(float('inf'))
            inter_bond_dist_error_list.append(float('inf'))
        else:
            # pick the first result of smina output (the best for minimizedAffinity)
            pred_mol = Chem.SDMolSupplier(res_filename)[0]
            tar_mol = Chem.SDMolSupplier(tar_filename)[0]
            
            # renumber atoms
            try:
                match = pred_mol.GetSubstructMatch(tar_mol)
                assert len(match) == pred_mol.GetNumAtoms()
            except:
                failed_pdb.append(pdb_id)
                rmsd_list.append(float('inf')); reorder_rmsd_list.append(float('inf'))
                inter_bond_dist_error_list.append(float('inf'))
                continue
            
            Chem.MolToPDBFile(pred_mol, reorder_out_filename)
            Chem.MolToPDBFile(tar_mol, reorder_tar_filename)
            
            pred_mol = Chem.RenumberAtoms(pred_mol, match)
            
            # calc rmsd
            assert len(pred_mol.GetConformers()) == 1
            assert len(tar_mol.GetConformers()) == 1
            pred_coords = pred_mol.GetConformers()[0].GetPositions()
            tar_coords = tar_mol.GetConformers()[0].GetPositions()
            pred_atoms = [atom.GetSymbol() for atom in pred_mol.GetAtoms()]
            tar_atoms = [atom.GetSymbol() for atom in tar_mol.GetAtoms()]
            assert pred_atoms == tar_atoms
            _rmsd = rmsd_func(tar_coords, pred_coords, mol=tar_mol)
            reorder_rmsd_res = reorder_rmsd(reorder_out_filename, reorder_tar_filename)
            reorder_rmsd_list.append(reorder_rmsd_res)
            
            name, resName, chainID, resSeq = row['bond'].split('-')[0].split(' ') # SG CYS A 145-C8 T9P A 405
            bond_ligand_atom, bond_ligand_het,bond_ligand_chain,bond_ligand_chain_idx = row['bond'].split('-')[1].split(' ')
            # aa part
            pocket_obmol, _, pocket_atoms, pocket_coords, pocket_bonds = parse_mol(pocket_filename, remove_H=True)
            bond_pocket_idx = _find_pocket_bond_idx(pocket_filename, [name,resName,chainID,resSeq])
            assert bond_pocket_idx != None
            assert pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue().GetName() == resName,f'Expect residue {resName} but got {pocket_obmol.GetAtom(bond_pocket_idx+1).GetResidue().GetName()}'
            
            # ligand part
            with open(tar_pdb_filename) as fp:
                pdb_mol = get_pdb_mol(fp.read(), tar_mol, [bond_ligand_atom, bond_ligand_het, bond_ligand_chain, bond_ligand_chain_idx])
            
            assert pdb_mol != None

            # renumber atoms
            match = pdb_mol.GetSubstructMatch(tar_mol)
            assert len(match) == pdb_mol.GetNumAtoms()
            pdb_mol = Chem.RenumberAtoms(pdb_mol, match)
            
            pdb_atoms = [atom.GetSymbol() for atom in pdb_mol.GetAtoms()]
            assert pdb_atoms == tar_atoms
            
            bond_ligand_idx = None
            # get bond ligand idx
            for i in range(pdb_mol.GetNumAtoms()):
                atom_name = pdb_mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()
                if atom_name == bond_ligand_atom:
                    bond_ligand_idx = i
                    break
            assert bond_ligand_idx != None
            
            assert tar_atoms[bond_ligand_idx][0] == bond_ligand_atom[0]
            inter_bond_dist_error = calc_inter_bond_dist_error(pred_coords, pocket_coords, tar_coords, [bond_pocket_idx, bond_ligand_idx])
            
            print(f'pdb_id:{pdb_id}-RMSD:{_rmsd}-InterBondDistError:{inter_bond_dist_error}')
            rmsd_list.append(_rmsd)
            inter_bond_dist_error_list.append(inter_bond_dist_error)
            
    print(f'failed pdb: {len(failed_pdb)}')
    print(failed_pdb)
    
    rmsd = np.array(rmsd_list)
    inter_bond_dist_error = np.array(inter_bond_dist_error_list)
    res_table = print_results(rmsd, inter_bond_dist_error)
    print('reorder result')
    reorder_res_table = print_results(np.array(reorder_rmsd_list), inter_bond_dist_error) # reorder rmsd
    
    if args.use_wandb:
        import wandb
        wandb.init(
            project="res_dock",
            config = json.load(open(args.dock_log_filename))
        )
        wandb.config.update(args)
        log_dict = {
            'res': res_table,
            'reorder_res': reorder_res_table,
        }
        wandb.log(log_dict)
