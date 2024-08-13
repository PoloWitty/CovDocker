# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import glob
import argparse
from docking_utils import (
    docking_data_pre,
    ensemble_iterations,
    print_results,
    rmsd_func,
    calc_inter_bond_dist_error
)
import warnings
from rmsd import NAMES_ELEMENT, ELEMENT_NAMES, reorder_hungarian, reorder_distance, reorder_inertia_hungarian
warnings.filterwarnings(action="ignore")
import pdb

def result_log(dir_path):
    ### result logging ###
    output_dir = os.path.join(dir_path, "cache")
    rmsd_results = []; reorder_rmsd_list = []; pdb_ids = []
    inter_bond_dist_results = []
    inter_bond_dist_list = []; inter_bond_dist_target_list = []; time_cost_list = []
    for path in glob.glob(os.path.join(output_dir, "*.pkl")):
        if path.endswith('.docking.pkl'):
            continue
        pdb_id = path.split("/")[-1].split(".")[0]
        pdb_ids.append(pdb_id)
        docking_result_path = path.replace(".pkl", ".docking.pkl")
        if os.path.exists(docking_result_path) is False: # skip failed docking result
            rmsd_results.append(float('inf'))
            reorder_rmsd_list.append(float('inf'))
            inter_bond_dist_results.append(float('inf'))
            inter_bond_dist_list.append(float('inf'))
            inter_bond_dist_target_list.append(float('inf'))
            # time_cost_list.append(time_cost) # only record time cost for successful docking
            continue
        (
            mol,
            bst_predict_coords,
            holo_coords,
            bst_loss,
            smi,
            pocket,
            pocket_coords,
            inter_bond,
            time_cost
        ) = pd.read_pickle(docking_result_path)
        rmsd = rmsd_func(holo_coords, bst_predict_coords, mol=mol)
        reorder_rmsd_res = reorder_rmsd(path.replace('.pkl','.ligand.xyz'),path.replace('.pkl','.tar.xyz'))
        inter_bond_dist_error, inter_bond_dist, inter_bond_dist_target = calc_inter_bond_dist_error(bst_predict_coords, pocket_coords, holo_coords, inter_bond, return_dist=True)
        rmsd_results.append(rmsd)
        reorder_rmsd_list.append(reorder_rmsd_res)
        inter_bond_dist_results.append(inter_bond_dist_error)
        inter_bond_dist_list.append(inter_bond_dist)
        inter_bond_dist_target_list.append(inter_bond_dist_target)
        time_cost_list.append(time_cost)

    pd.DataFrame({
        'pdb_id': pdb_ids,
        'rmsd': rmsd_results,
        'reorder_rmsd': reorder_rmsd_list,
        'rmsd(ib)': inter_bond_dist_results,
        'inter_bond_dist': inter_bond_dist_list,
        'inter_bond_dist_target': inter_bond_dist_target_list
    }).to_csv(os.path.join(dir_path, 'dock_res.csv'), index=False)
    rmsd_results = np.array(rmsd_results)
    reorder_rmsd_list = np.array(reorder_rmsd_list)
    inter_bond_dist_results = np.array(inter_bond_dist_results)
    time_cost_list = np.array(time_cost_list)
    res_table = print_results(rmsd_results, inter_bond_dist_results)
    reorder_res_table = print_results(reorder_rmsd_list, inter_bond_dist_results)
    time_cost = {
        'Average dock time cost': np.mean(time_cost_list),
        'Total dock time cost': np.sum(time_cost_list)
    }
    print(str(time_cost))
    return res_table,reorder_res_table,time_cost


def get_info_from_xyz_file(pdb_filename, sym2idx):
    coords = []; syms = []
    with open(pdb_filename, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]: # skip number and \n
            line = line.strip()
            if not line.startswith('H'):
                sym = line[:3].strip().title(); x=line[3:15]; y=line[15:27]; z=line[27:39]
                coord = [float(x), float(y), float(z)]
                coords.append(coord)
                syms.append(sym2idx[sym]) # convert to index
    return np.array(syms), np.array(coords)

def calculate_rmsd(P, Q):
    """Calculate the RMSD between two aligned point clouds."""
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))


def reorder_rmsd(pred_filename, target_filename):
    assert pred_filename.endswith('.xyz') and target_filename.endswith('.xyz'), 'Only support xyz file'
    P_atoms, P_coords = get_info_from_xyz_file(target_filename, NAMES_ELEMENT)
    Q_atoms, Q_coords = get_info_from_xyz_file(pred_filename, NAMES_ELEMENT)
    
    # reorder the atoms and coordinates
    Q_reordered = reorder_hungarian(P_atoms, Q_atoms, P_coords, Q_coords) # Align the principal intertia axis and then re-orders the input atom list and xyz coordinates using the Hungarian method (using optimized column results)
    Q_atoms = Q_atoms[Q_reordered]
    Q_coords = Q_coords[Q_reordered]
    reordered_rmsd = calculate_rmsd(P_coords, Q_coords)
    return reordered_rmsd

def write_xyz_file(atoms, coords, output_filename):
    with open(output_filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom:3s}{coord[0]:12.6f}{coord[1]:12.6f}{coord[2]:12.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="docking")
    parser.add_argument(
        "--reference-file",
        type=str,
        default="./protein_ligand_binding_pose_prediction/test.lmdb",
        help="Location of the reference set",
    )
    parser.add_argument("--nthreads", type=int, default=40, help="num of threads")
    parser.add_argument(
        "--predict-file",
        type=str,
        default="./infer_pose/save_pose_test.out.pkl",
        help="Location of the prediction file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./protein_ligand_binding_pose_prediction",
        help="Location of the docking output path",
    )
    parser.add_argument(
        "--optimization-model",
        type=str,
        default="coordinate",
        help="Optimize coordinates ('coordinate') or ligand internal torsions ('conformer')",
        choices=["coordinate", "conformer"],
    )
    parser.add_argument(
        "--noncovalent",
        type=int,
        default=1,
        help="Use noncovalent docking method to process data",
    )
    parser.add_argument("--do-post-process",type=int, default=0)
    parser.add_argument("--run-id", type=str, default='default-run-id')
    parser.add_argument("--use-wandb",type=int, default=0, help='whether use wandb to store result')
    parser.add_argument("--infer-config-filename", type=str, default='' ,help="infer config filename to log")
    parser.add_argument("--infer-time-filename", type=str, default='', help="stored infer time filename")
    parser.add_argument("--pipe-dock",type=int, default=0, help='whether using reactive site model predicted pocket to dock')
    parser.add_argument("--do-statistics-on-auxiliary-loss", type=int, default=0, help='whether do statistics on bond length for auxiliary loss')
    args = parser.parse_args()

    # result_log(args.output_path)
    # breakpoint()

    raw_data_path, predict_path, dir_path, nthreads, model_choice = (
        args.reference_file,
        args.predict_file,
        args.output_path,
        args.nthreads,
        args.optimization_model,
    )
    tta_times = 1
    (
        mol_list,
        smi_list,
        pocket_list,
        pocket_coords_list,
        distance_predict_list,
        holo_distance_predict_list,
        holo_coords_list,
        holo_center_coords_list,
        inter_bond_list,
        holo_atom_list
    ) = docking_data_pre(raw_data_path, predict_path, args.noncovalent, args.do_statistics_on_auxiliary_loss)
    iterations = ensemble_iterations(
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
        tta_times=tta_times,
    )
    sz = len(mol_list) // tta_times
    new_pocket_list = pocket_list[::tta_times]
    output_dir = os.path.join(dir_path, "cache")
    os.makedirs(output_dir, exist_ok=True)

    def dump(content):
        pocket = content[3]
        output_name = os.path.join(output_dir, "{}.pkl".format(pocket))
        try:
            os.remove(output_name)
        except:
            pass
        pd.to_pickle(content, output_name)
        holo_coords_list = content[-4]; holo_center_coords_list = content[-3]; holo_atom_list = content[-1]
        # tar_coords_list = [[coord[i].item() + center_coord[i].item() for i in range(3)] for coord, center_coord in zip(holo_coords_list,holo_center_coords_list)]
        tar_coords_list = holo_coords_list + holo_center_coords_list.numpy()
        write_xyz_file(holo_atom_list, tar_coords_list, output_name.replace('.pkl','.tar.xyz'))
        return True

    # skip step if repeat
    with Pool(nthreads) as pool:
        for inner_output in tqdm(pool.imap_unordered(dump, iterations), total=sz):
            if not inner_output:
                print("fail to dump")

    def single_docking(pocket_name):
        input_name = os.path.join(output_dir, "{}.pkl".format(pocket_name))
        output_name = os.path.join(output_dir, "{}.docking.pkl".format(pocket_name))
        output_ligand_name = os.path.join(output_dir, "{}.ligand.sdf".format(pocket_name))
        try:
            os.remove(output_name)
        except:
            pass
        try:
            os.remove(output_ligand_name)
        except:
            pass
        cmd = "python ./unimol/utils/{}_model.py --input {} --output {} --output-ligand {} --do-post-process {}".format(model_choice, input_name, output_name, output_ligand_name, int(args.do_post_process))
        os.system(cmd)
        return True

    with Pool(nthreads) as pool:
        for inner_output in tqdm(
            pool.imap_unordered(single_docking, new_pocket_list), total=len(new_pocket_list)
        ):
            if not inner_output:
                print("fail to docking")

    res_table,reorder_res_table,time_cost = result_log(args.output_path)
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
            'dock_time': time_cost
        }
        wandb.log(log_dict)
