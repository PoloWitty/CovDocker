# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
import warnings

warnings.filterwarnings(action="ignore")
from rdkit.Chem import rdMolTransforms
import copy
import lmdb
import pickle
import pandas as pd
import tqdm
import torch
from typing import Dict, List, Optional
from conf_gen_cal_metrics import clustering, single_conf_gen
from tabulate import tabulate

mol_dictionary = ["[PAD]","[CLS]","[SEP]","[UNK]","C","N","O","S","H","Cl","F","Br","I","Si","P","B","Na","K","Al","Ca","Sn","As","Hg","Fe","Zn","Cr","Se","Gd","Au","Li"]

def add_all_conformers_to_mol(mol: Chem.Mol, conformers: List[np.ndarray]) -> Chem.Mol:
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    for i, conf_pos in enumerate(conformers):
        conf = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conf, assignId=True)

        conf = mol.GetConformer(i)
        positions = conf_pos.tolist()
        for j in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(j, positions[j])
    return mol


def get_torsions(m: Chem.Mol, removeHs=True) -> List:
    if removeHs:
        m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                    m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList


def load_lmdb_data(lmdb_path, key):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    _keys = list(txn.cursor().iternext(values=False))
    collects = []
    for idx in range(len(_keys)):
        datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        collects.append(data[key])
    return collects


def reprocess_content(content: Dict, base_mol: Optional[Chem.Mol] = None, M: int = 2000, N: int = 10, mmff: bool = False, seed: int = 42, stereo_from3d: bool = True) -> Dict:
    """ Reprocess a data point in the LMDB schema for Docking usage. Ensures correct stereochemistry.
    Basic principle is to perceive stereochem from label molecule's 3D and keep it intact.
    Use default values for best results

    Args:
        content: A dictionary of the LMDB schema. (atoms, holo_mol, mol_list, cooredinates, etc.)
        base_mol: The molecule to replace the holo_mol with, if passed
        M: The number of conformers to generate
        N: The number of clusters to group conformers and pick a representative from
        mmff: Whether to use MMFF minimization after conformer generation
        seed: The random seed to use for conformer generation
        stereo_from3d: Whether to perceive stereochemistry from the 3D coordinates of the label molecule

    Returns:
        A copy of the original, with the holo_mol replaced with the base_mol, and coordinates added.
    """
    if base_mol is None:
        base_mol = content["holo_mol"]
    # Copy so we don't change inputs
    content = copy.deepcopy(content)
    base_mol = copy.deepcopy(base_mol)
    base_mol = Chem.AddHs(base_mol, addCoords=True)
    # assign stereochem from 3d
    if stereo_from3d and base_mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(base_mol)
    ori_smiles = Chem.MolToSmiles(base_mol)
    # create new, clean molecule
    remol = Chem.MolFromSmiles(ori_smiles)
    # reorder to match and add Hs
    idxs = remol.GetSubstructMatches(Chem.RemoveHs(base_mol))
    if isinstance(idxs[0], tuple):
        idxs = idxs[0]
    idxs = list(map(int, idxs))
    remol = Chem.RenumberAtoms(remol, idxs)
    remol = Chem.AddHs(remol, addCoords=True)
    # overwrite - write the diverse conformer set for potential later reuse
    content["coordinates"] = [x for x in clustering(remol, M=M, N=N, seed=seed, removeHs=False, mmff=mmff)]
    content["mol_list"] = [
        Chem.AddHs(
            copy.deepcopy(add_all_conformers_to_mol(
                Chem.RemoveHs(remol), content["coordinates"]
            )), addCoords=True
        ) for i in range(N)
    ]
    content["holo_mol"] = copy.deepcopy(base_mol)
    content["atoms"] = [a.GetSymbol() for a in base_mol.GetAtoms()]
    return content

def set_coord(mol, coords):
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, coords[i].tolist())
    return mol

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except KeyboardInterrupt:
        exit()
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol

# modified to satisfy covDocker's data
def docking_data_pre(raw_data_path, predict_path, use_noncovalent=False, do_statistics_on_auxiliary_loss=False):
    mol_list = load_lmdb_data(raw_data_path, "mol")
    inter_bond_list = load_lmdb_data(raw_data_path, "inter_bond")
    pdb_ids = load_lmdb_data(raw_data_path, "pocket")
    for idx,mol in tqdm.tqdm(enumerate(mol_list),desc='process mol',total=len(mol_list)):
        mol = Chem.RemoveHs(mol)
        mol.RemoveAllConformers()
        mol = generate_conformation(mol)
        mol_list[idx] = mol
    predict = pd.read_pickle(predict_path)
    dict_list = ['[PAD]','[CLS]','[SEP]','[UNK]'] + [5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]
    (
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        holo_atom_list,
    ) = ([], [], [], [], [], [], [], [])
    idx = 0
    is_min_dist = []
    is_min_dist_gt = []
    for batch in tqdm.tqdm(predict, desc="Prepare data"):
        sz = batch["atoms"].size(0)
        for i in range(sz):
            smi = batch["smi_name"][i]
            smi_list.append(smi)
            pocket_list.append(batch["pocket_name"][i])

            if not use_noncovalent:
                docking_src_tokens = batch["atoms"][i]
                eos_indices = torch.where(docking_src_tokens==2)[0]
                ligand_eos_idx = eos_indices[0]
                pocket_eos_idx = eos_indices[1]
                mol_sz = ligand_eos_idx + 1 -2
                pocket_sz = pocket_eos_idx - ligand_eos_idx -2 
                
                ligand_mask = torch.zeros_like(docking_src_tokens).to(torch.bool)
                ligand_mask[1:ligand_eos_idx] = True
                pocket_mask = torch.zeros_like(docking_src_tokens).to(torch.bool)
                pocket_mask[ligand_eos_idx+2:pocket_eos_idx] = True
                # pred
                distance_predict = batch["holo_distance_predict"][i]
                mol = mol_list[idx]
                mol_atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
                model_atoms = [dict_list[a] for a in docking_src_tokens[ligand_mask].tolist() ]
                assert mol_atoms==model_atoms, f'atoms: expected {mol_atoms}, got {model_atoms}'
                
                cross_distance_predict = (distance_predict[ligand_mask][:, pocket_mask] + distance_predict[pocket_mask][:, ligand_mask].T)/2
                ligand_distance_predict = distance_predict[ligand_mask][:, ligand_mask]
                
                # target
                coords = batch["holo_coordinates"][i]
                pocket_coords = coords[pocket_mask, :]
                ligand_coords = coords[ligand_mask, :]
                
                assert ligand_coords.size(0) == mol_sz, f'ligand coords: expected {mol_sz}, got {coords.size(0)}'
                assert pocket_coords.size(0) == pocket_sz, f'pocket coords: expected {pocket_sz}, got {pocket_coords.size(0)}'
                assert cross_distance_predict.shape == (mol_sz, pocket_sz), f'crosss dist: expected {(mol_sz, pocket_sz)}, got {cross_distance_predict.shape}'
                
                holo_center_coordinates = batch["holo_center_coordinates"][i][:3]
                pocket_coords = pocket_coords.numpy().astype(np.float32)
                distance_predict = cross_distance_predict.numpy().astype(np.float32)
                holo_distance_predict = ligand_distance_predict.numpy().astype(np.float32)
                # Fill diagonal with 0, issue with the model not learning to predict 0 distance
                np.fill_diagonal(holo_distance_predict, 0)
                holo_coords = ligand_coords.numpy().astype(np.float32)
            else:
                distance_predict = batch["cross_distance_predict"][i]
                ligand_atoms = batch["atoms"][i]
                token_mask = batch["atoms"][i] > 2
                pocket_token_mask = batch["pocket_atoms"][i] > 2
                distance_predict = distance_predict[token_mask][:, pocket_token_mask]
                distance_target = batch["cross_distance_target"][i]
                distance_target = distance_target[token_mask][:, pocket_token_mask]
                
                pocket_coords = batch["pocket_coordinates"][i]
                pocket_coords = pocket_coords[pocket_token_mask, :]
                orig2cropped = batch['orig2cropped_pocket'][i]
                # map the orig index to cropped index
                orig = inter_bond_list[idx][0]
                inter_bond_list[idx][0] = orig2cropped[orig].item()
                try:
                    assert inter_bond_list[idx][0]!=-1
                except:
                    breakpoint()

                inter_bond_dist = distance_predict[inter_bond_list[idx][1], inter_bond_list[idx][0]]
                if inter_bond_dist == distance_predict.min():
                    is_min_dist.append(1)
                else:
                    is_min_dist.append(0)
                inter_bond_dist_gt = distance_target[inter_bond_list[idx][1], inter_bond_list[idx][0]]
                if inter_bond_dist_gt == distance_target.min():
                    is_min_dist_gt.append(1)
                else:
                    is_min_dist_gt.append(0)
                holo_distance_predict = batch["holo_distance_predict"][i]
                holo_distance_predict = holo_distance_predict[token_mask][:, token_mask]

                holo_coordinates = batch["holo_coordinates"][i]
                holo_coordinates = holo_coordinates[token_mask, :]
                ligand_atoms = ligand_atoms[token_mask].tolist()
                ligand_atoms = [mol_dictionary[atom] for atom in ligand_atoms]

                holo_center_coordinates = batch["holo_center_coordinates"][i][:3]

                pocket_coords = pocket_coords.numpy().astype(np.float32)
                distance_predict = distance_predict.numpy().astype(np.float32)
                holo_distance_predict = holo_distance_predict.numpy().astype(np.float32)
                # Fill diagonal with 0, issue with the model not learning to predict 0 distance
                np.fill_diagonal(holo_distance_predict, 0)
                #
                holo_coords = holo_coordinates.numpy().astype(np.float32)
            
            pocket_coords_list.append(pocket_coords)
            distance_predict_list.append(distance_predict)
            holo_distance_predict_list.append(holo_distance_predict)
            holo_coords_list.append(holo_coords)
            holo_center_coords_list.append(holo_center_coordinates)
            holo_atom_list.append(ligand_atoms)
            idx += 1
    # breakpoint()
    if do_statistics_on_auxiliary_loss:
        print(f'is_min_dist: {sum(is_min_dist)}; is_min_dist_gt: {sum(is_min_dist_gt)}')
        is_min = {'pdb_id':pdb_ids, 'is_min_dist':is_min_dist, 'is_min_dist_gt':is_min_dist_gt}
        pd.DataFrame(is_min).to_csv(os.path.dirname(predict_path) + '/infer_res.csv')
    assert pocket_list == pdb_ids, 'Expect data in test.lmdb to be in the same order as the model input'
    return (
        mol_list, # 3
        
        smi_list,
        pocket_list,
        pocket_coords_list, # 1
        distance_predict_list, # 2
        holo_distance_predict_list, # 2
        holo_coords_list, # 1
        holo_center_coords_list,
        inter_bond_list,
        holo_atom_list
    )


def ensemble_iterations(
    mol_list,
    smi_list,
    pocket_list,
    pocket_coords_list,
    distance_predict_list,
    holo_distance_predict_list,
    holo_coords_list,
    holo_center_coords_list,
    inter_bond_list,
    holo_atom_list,
    tta_times=10,
    seed=42,
):
    sz = len(mol_list)
    for i in range(sz // tta_times):
        start_idx, end_idx = i * tta_times, (i + 1) * tta_times
        distance_predict_tta = distance_predict_list[start_idx:end_idx]
        holo_distance_predict_tta = holo_distance_predict_list[start_idx:end_idx]

        mol = copy.deepcopy(mol_list[start_idx])
        # rdkit_mol = single_conf_gen(mol, num_confs=tta_times, seed=seed)
        rdkit_mol = mol
        sz = len(rdkit_mol.GetConformers())
        initial_coords_list = [
            rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            for i in range(sz)
        ]

        yield [
            initial_coords_list,
            mol,
            smi_list[start_idx],
            pocket_list[start_idx],
            pocket_coords_list[start_idx],
            distance_predict_tta,
            holo_distance_predict_tta,
            holo_coords_list[start_idx],
            holo_center_coords_list[start_idx],
            inter_bond_list[start_idx],
            holo_atom_list[start_idx]
        ]

def calc_inter_bond_dist_error(predict_coords, pocket_coords, holo_coords, inter_bond, return_dist=False):
    inter_bond_dist = np.linalg.norm(predict_coords[inter_bond[1]] - pocket_coords[inter_bond[0]], 2)
    inter_bond_dist_target = np.linalg.norm(holo_coords[inter_bond[1]] - pocket_coords[inter_bond[0]], 2)
    inter_bond_dist_error = np.sqrt((inter_bond_dist_target - inter_bond_dist)**2)
    if return_dist:
        return inter_bond_dist_error, inter_bond_dist, inter_bond_dist_target    
    else:
        return inter_bond_dist_error

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