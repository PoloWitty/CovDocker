# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import pandas as pd
from rdkit import Chem
import pickle
import argparse
from docking_utils import rmsd_func, calc_inter_bond_dist_error
import warnings
import numpy as np
import time

warnings.filterwarnings(action="ignore")


def single_SF_loss(
    predict_coords,
    pocket_coords,
    distance_predict,
    holo_distance_predict,
    inter_bond,
    dist_threshold=4.5, 
    use_min_dist_norm_loss=False
):
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = (
        (dist[distance_mask] - distance_predict[distance_mask]) ** 2
    ).mean()
    dist_score = ((holo_dist - holo_distance_predict) ** 2).mean()
    
    if use_min_dist_norm_loss:
        inter_bond_dist = dist[inter_bond[1],inter_bond[0]]
        inter_bond_dist = inter_bond_dist.unsqueeze(-1).unsqueeze(-1)
        min_dist_norm_loss =  torch.nn.functional.relu((inter_bond_dist - dist)[distance_mask])
        min_dist_norm_loss = min_dist_norm_loss.mean()
    else:
        min_dist_norm_loss = torch.zeros_like(cross_dist_score)
    
    loss = cross_dist_score * 1.0 + dist_score * 5.0 + min_dist_norm_loss
    return loss


def scoring(
    predict_coords,
    pocket_coords,
    distance_predict,
    holo_distance_predict,
    inter_bond,
    dist_threshold=4.5,
    use_min_dist_norm_loss=False
):
    predict_coords = predict_coords.detach()
    dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    holo_dist = torch.norm(
        predict_coords.unsqueeze(1) - predict_coords.unsqueeze(0), dim=-1
    )
    distance_mask = distance_predict < dist_threshold
    cross_dist_score = (
        (dist[distance_mask] - distance_predict[distance_mask]) ** 2
    ).mean()
    dist_score = ((holo_dist - holo_distance_predict) ** 2).mean()
    
    if use_min_dist_norm_loss:
        inter_bond_dist = dist[inter_bond[1],inter_bond[0]]
        inter_bond_dist = inter_bond_dist.unsqueeze(-1).unsqueeze(-1)
        min_dist_norm_loss =  torch.nn.functional.relu((inter_bond_dist - dist)[distance_mask])
        min_dist_norm_loss = min_dist_norm_loss.mean()
    else:
        min_dist_norm_loss = torch.zeros_like(cross_dist_score)
    return cross_dist_score.numpy(), dist_score.numpy(), min_dist_norm_loss.numpy()

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

def dock_with_gradient(
    coords,
    pocket_coords,
    distance_predict_tta,
    holo_distance_predict_tta,
    loss_func=single_SF_loss,
    holo_coords=None,
    inter_bond=None,
    iterations=20000,
    early_stoping=5,
    do_post_process = 0
):
    bst_loss, bst_coords, bst_meta_info = 10000.0, coords, None
    for i, (distance_predict, holo_distance_predict) in enumerate(
        zip(distance_predict_tta, holo_distance_predict_tta)
    ):
        new_coords = copy.deepcopy(coords)
        _coords, _loss, _meta_info = single_dock_with_gradient(
            new_coords,
            pocket_coords,
            distance_predict,
            holo_distance_predict,
            loss_func=loss_func,
            holo_coords=holo_coords,
            inter_bond=inter_bond,
            iterations=iterations,
            early_stoping=early_stoping,
        )
        if bst_loss > _loss:
            bst_coords = _coords
            bst_loss = _loss
            bst_meta_info = _meta_info
    if do_post_process:
        post_process_coords(bst_coords, pocket_coords, inter_bond, holo_coords ,move_type=do_post_process)
    return bst_coords, bst_loss, bst_meta_info


def single_dock_with_gradient(
    coords,
    pocket_coords,
    distance_predict,
    holo_distance_predict,
    loss_func=single_SF_loss,
    holo_coords=None,
    inter_bond=None,
    iterations=20000,
    early_stoping=5,
):
    coords = torch.from_numpy(coords).float()
    pocket_coords = torch.from_numpy(pocket_coords).float()
    distance_predict = torch.from_numpy(distance_predict).float()
    holo_distance_predict = torch.from_numpy(holo_distance_predict).float()

    if holo_coords is not None:
        holo_coords = torch.from_numpy(holo_coords).float()

    coords.requires_grad = True
    optimizer = torch.optim.LBFGS([coords], lr=1.0)
    bst_loss, times = 10000.0, 0
    for i in range(iterations):

        def closure():
            optimizer.zero_grad()
            loss = loss_func(
                coords, pocket_coords, distance_predict, holo_distance_predict, inter_bond
            )
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        if loss.item() < bst_loss:
            bst_loss = loss.item()
            times = 0
        else:
            times += 1
            if times > early_stoping:
                break

    meta_info = scoring(coords, pocket_coords, distance_predict, holo_distance_predict, inter_bond)
    return coords.detach().numpy(), loss.detach().numpy(), meta_info


def set_coord(mol, coords):
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, coords[i].tolist())
    return mol


def add_coord(mol, xyz):
    x, y, z = xyz
    conf = mol.GetConformer(0)
    pos = conf.GetPositions()
    pos[:, 0] += x
    pos[:, 1] += y
    pos[:, 2] += z
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(
            i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
        )
    return mol



def single_docking(input_path, output_path, output_ligand_path, do_post_process=0):
    content = pd.read_pickle(input_path)
    (
        init_coords_tta,
        mol,
        smi,
        pocket,
        pocket_coords,
        distance_predict_tta,
        holo_distance_predict_tta,
        holo_coords,
        holo_cener_coords,
        inter_bond,
        holo_atom_list
    ) = content
    sample_times = len(init_coords_tta)
    bst_predict_coords, bst_loss, bst_meta_info = None, 1000.0, None
    start_time = time.time()
    for i in range(sample_times):
        init_coords = init_coords_tta[i]
        predict_coords, loss, meta_info = dock_with_gradient(
            init_coords,
            pocket_coords,
            distance_predict_tta,
            holo_distance_predict_tta,
            holo_coords=holo_coords,
            inter_bond=inter_bond,
            loss_func=single_SF_loss,
            do_post_process=do_post_process
        )
        if loss < bst_loss:
            bst_loss = loss
            bst_predict_coords = predict_coords
            bst_meta_info = meta_info
    end_time = time.time()
    time_cost = end_time - start_time
    _rmsd = round(rmsd_func(holo_coords, bst_predict_coords, mol), 4)
    _cross_score = round(float(bst_meta_info[0]), 4)
    _self_score = round(float(bst_meta_info[1]), 4)
    _inter_bond_dist_error = round(calc_inter_bond_dist_error(bst_predict_coords, pocket_coords, holo_coords, inter_bond), 4)
    print(f"{pocket}-{smi}-RMSD:{_rmsd}-{_inter_bond_dist_error}-{_cross_score}-{_self_score}-{time_cost}")
    mol = Chem.RemoveHs(mol)
    mol = set_coord(mol, bst_predict_coords)

    if output_path is not None:
        with open(output_path, "wb") as f:
            pickle.dump(
                [mol, bst_predict_coords, holo_coords, bst_loss, smi, pocket, pocket_coords, inter_bond, time_cost],
                f,
            )
    if output_ligand_path is not None:
        mol = add_coord(mol, holo_cener_coords.numpy())
        Chem.MolToMolFile(mol, output_ligand_path)
        Chem.MolToXYZFile(mol, output_ligand_path.replace('.sdf','.xyz'))

    return True


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="Docking with gradient")
    parser.add_argument("--input", type=str, help="input file.")
    parser.add_argument("--output", type=str, default=None, help="output path.")
    parser.add_argument(
        "--output-ligand", type=str, default=None, help="output ligand sdf path."
    )
    parser.add_argument("--do-post-process",type=int, default=0)
    args = parser.parse_args()

    single_docking(args.input, args.output, args.output_ligand, do_post_process=args.do_post_process)
