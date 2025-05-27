"""
desc:	get docking result and upload to wandb
author:	Yangzhe Peng
date:	2025/02/22
"""


import lmdb
import pickle
import wandb
import tqdm
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
import argparse
import os
import json
from rmsd import reorder_hungarian
from docking_utils import rmsd_func, calc_inter_bond_dist_error, print_results

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

def calculate_rmsd(P, Q):
    """Calculate the RMSD between two aligned point clouds."""
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))


def reorder_rmsd(P_atoms, P_coords, Q_atoms, Q_coords):
    # reorder the atoms and coordinates
    Q_reordered = reorder_hungarian(P_atoms, Q_atoms, P_coords, Q_coords) # Align the principal intertia axis and then re-orders the input atom list and xyz coordinates using the Hungarian method (using optimized column results)
    Q_atoms = Q_atoms[Q_reordered]
    Q_coords = Q_coords[Q_reordered]
    reordered_rmsd = calculate_rmsd(P_coords, Q_coords)
    return reordered_rmsd


def RMSD(coord_predict, coord_target):
    mask = coord_target != 0 # pad, bos, eos
    rmsd = np.sqrt(np.sum(((coord_predict - coord_target) ** 2) * mask) / (mask[:,0].sum()))
    return rmsd

def post_process_coords(coords, pocket_coords, inter_bond, holo_coords, move_type=3):
    ib_p_coord = pocket_coords[inter_bond[0]] # inter-bond pocket atom coord
    ib_l_coord = coords[inter_bond[1]] # inter-bond ligand atom coord
    ib_l_coord_tar = holo_coords[inter_bond[1]] # inter-bond ligand atom coord in target
    move_vec = ib_p_coord - ib_l_coord
    
    # stat result for inter-bond len
    # count  2308.000000
    # mean      1.603774
    # std       0.223993
    
    # scale move_vec to ib_len
    ib_len_now = np.linalg.norm(move_vec)
    mean = 1.603774; var = 0.223993
    ib_len_tar = np.random.normal(mean, var, 1)
    ib_move_len = ib_len_now - ib_len_tar
    move_vec = move_vec / ib_len_now * ib_move_len
    if move_type==0: # do not post-process
        return
    elif move_type==1:
        if not (ib_len_now >= mean - 10*var and ib_len_now <= mean + 10*var): # only move those out of 10 sigma
            ib_l_coord += move_vec
    elif move_type==2: # move the whole ligand atoms
        if not (ib_len_now >= mean - 10*var and ib_len_now <= mean + 10*var): # only move those out of 10 sigma
            coords += move_vec
    elif move_type==3: # only move inter-bond ligand atom
        ib_l_coord += move_vec # operate will reflect on coords
    else:
        raise NotImplementedError

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Upload docking results to wandb')
    parser.add_argument('--predict_file', type=str, required=True, help='Path to the prediction file')
    parser.add_argument('--reference_file', type=str, required=True, help='Path to the reference file')
    parser.add_argument("--use-wandb",type=int, default=0, help='whether use wandb to store result')
    parser.add_argument("--infer-config-filename", type=str, default='' ,help="infer config filename to log")
    parser.add_argument("--infer-time-filename", type=str, default='', help="stored infer time filename")
    parser.add_argument("--pipe-dock",type=int, default=0, help='whether using reactive site model predicted pocket to dock')
    parser.add_argument("--post-process", type=int, default=0, help='whether post-process the coordinates')
    
    args = parser.parse_args()

    predict_file = args.predict_file
    reference_file = args.reference_file

    
    pdb_id_list = []
    rmsd_list = []; reorder_rmsd_list = []
    inter_bond_dist_error_list = []
    inter_bond_dist_list = []; inter_bond_dist_target_list = []
    
    
    inter_bond_list = load_lmdb_data(reference_file, 'inter_bond')
    assert len(inter_bond_list)==220, 'the number of input should be 220 under pipe docking scenario'
    pred_logs = pickle.load(open(predict_file, 'rb'))
    
    for i, log in enumerate(pred_logs):
        # move to cpu
        for key in log:
            if isinstance(log[key], torch.Tensor):
                log[key] = log[key].cpu().numpy()

        # load data from log
        coord_preds = log['coord_predict'].squeeze()
        coord_targets = log['coord_target'].squeeze()
        pocket_coords = log['pocket_coordinates'].squeeze()
        assert len(log['pocket_name']) == 1, 'Batch size should be 1'
        pdb_ids = log['pocket_name'][0]
        orig2cropped = log['orig2cropped_pocket'].squeeze()
        lig_atoms = log['atoms'].squeeze()
        
        # drop special tokens
        lig_mask = coord_targets != 0
        coord_preds = coord_preds[lig_mask].reshape(-1, coord_preds.shape[-1])
        coord_targets = coord_targets[lig_mask].reshape(-1, coord_targets.shape[-1])
        lig_atoms = lig_atoms[lig_mask[:,0]]
        poc_mask = pocket_coords != 0
        pocket_coords = pocket_coords[poc_mask].reshape(-1, pocket_coords.shape[-1])
        
        # map cropped idx to original idx
        orig = inter_bond_list[i][0]
        inter_bond_list[i][0] = orig2cropped[orig].item()
        if args.post_process:
            post_process_coords(coord_preds, pocket_coords, inter_bond_list[i], coord_targets, move_type=args.post_process)
        
        # calc rmsd
        rmsd_res = RMSD(coord_preds, coord_targets)
        reorder_rmsd_res = reorder_rmsd(lig_atoms, coord_preds, lig_atoms, coord_targets)
        
        # calc rmsd (inter bond)
        inter_bond_dist_error, inter_bond_dist, inter_bond_dist_target = calc_inter_bond_dist_error(coord_preds, pocket_coords, coord_targets, inter_bond_list[i], return_dist=True)
        
        
        # save res to list
        pdb_id_list.append(pdb_ids)
        rmsd_list.append(rmsd_res)
        reorder_rmsd_list.append(reorder_rmsd_res)
        inter_bond_dist_error_list.append(inter_bond_dist_error)
        inter_bond_dist_list.append(inter_bond_dist)
        inter_bond_dist_target_list.append(inter_bond_dist_target)
    
    pd.DataFrame({
        'pdb_id': pdb_id_list,
        'rmsd': rmsd_list,
        'reorder_rmsd': reorder_rmsd_list,
        'rmsd(ib)': inter_bond_dist_error_list,
        'inter_bond_dist': inter_bond_dist_list,
        'inter_bond_dist_target': inter_bond_dist_target_list
    }).to_csv(os.path.join(os.path.dirname(predict_file), 'dock_res.csv'), index=False)
    
    rmsd_res = np.array(rmsd_list)
    reorder_rmsd_res = np.array(reorder_rmsd_list)
    inter_bond_dist_error_res = np.array(inter_bond_dist_error_list)
    res_table = print_results(rmsd_res, inter_bond_dist_error_res)
    reorder_res_table = print_results(reorder_rmsd_res, inter_bond_dist_error_res)
    
    if args.use_wandb:
        import wandb
        wandb.init(
            project = 'res_pipe_dock' if args.pipe_dock else 'res_dock',
            config = json.load(open(args.infer_config_filename))
        )
        wandb.config.update(args)
        infer_time = json.load(open(args.infer_time_filename))
        log_dict = {
            'res': res_table,
            'reorder_res': reorder_res_table,
            'infer_time': infer_time,
        }
        wandb.log(log_dict)
