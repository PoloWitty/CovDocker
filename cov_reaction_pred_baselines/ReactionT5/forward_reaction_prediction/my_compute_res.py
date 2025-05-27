"""
desc:	compute topK molecular exact match accuracy result for ReactionT5
author:	Yangzhe Peng
date:	2024/05/12
"""

import argparse
import pandas as pd
import numpy as np
import tqdm
from typing import Tuple, List
from rdkit import Chem
import json

# copy from Chemformer/molbart/modules/decoder.py BeamSearchSampler._compute_accuracy
def _compute_accuracy(
    sampled_smiles: List[List[str]], target_smiles: List[str], top_Ks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computing top-K accuracy for each K in 'top_Ks'.
    """

    n_beams = np.max(
        np.array(
            [1, np.max(np.asarray([len(smiles) for smiles in sampled_smiles]))]
        )
    )
    top_Ks = top_Ks[top_Ks <= n_beams]
    n_Ks = len(top_Ks)

    accuracy = np.zeros(n_Ks)

    is_in_set = np.zeros((len(sampled_smiles), n_Ks), dtype=bool)
    for i_k, K in enumerate(top_Ks):
        for i_sample, mols in enumerate(sampled_smiles):
            top_K_mols = mols[0:K]

            if len(top_K_mols) == 0:
                continue
            is_in_set[i_sample, i_k] = target_smiles[i_sample] in top_K_mols

    is_in_set = np.cumsum(is_in_set, axis=1)
    accuracy = np.mean(is_in_set > 0, axis=0)
    return accuracy, top_Ks

def _canonicalize_smiles(input_smiles: str) -> str:
    """
    Canonicalize smiles and sort the (possible) multiple molcules.

    Args:
        input_smiles (str): SMILES string.
    Returns:
        str: Canonicalized SMILES string.
    """
    mol = Chem.MolFromSmiles(input_smiles)
    if mol is None:
        return input_smiles
    smiles_canonical = Chem.MolToSmiles(mol)

    smiles_sep = np.array(smiles_canonical.split("."))
    smiles_canonical = ".".join(np.sort(smiles_sep))
    return smiles_canonical

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer_file", 
        type=str, 
        required=True, 
        help="The path to data used for computing metrics"
    )
    parser.add_argument(
        "--ref_file",
        type=str,
        required=True,
        help="The path to reference data"
    )
    parser.add_argument(
        "--topK",
        type=int,
        default=10,
        help='Top K accuracy to compute. Default is 10.'
    )
    parser.add_argument("--use-wandb",type=int, default=0, help='whether use wandb to store result')
    parser.add_argument("--infer-config-filename", type=str, default='' ,help="infer config filename to log")
    parser.add_argument("--run-id", type=str, default='default-run-id')
    args = parser.parse_args()
    
    infer_df = pd.read_csv(args.infer_file)
    ref_df = pd.read_csv(args.ref_file)
    assert len(infer_df)==len(ref_df)
    
    target_smiles = []
    sampled_smiles = []
    for i in tqdm.trange(len(ref_df),desc='gather res'):
        infer_row = infer_df.iloc[i]
        ref_row = ref_df.iloc[i]
        
        ref_reactant = ref_row['REACTANT']
        infer_reactant = infer_row['input'].replace('REACTANT:','').replace('REAGENT: ','')
        assert ref_reactant == infer_reactant
        
        ref_product = ref_row['PRODUCT']
        infer_res = []
        for k in range(args.topK):
            k_res = infer_row[f'{k}th']
            infer_res.append(k_res)
        sampled_smiles.append(infer_res)
        target_smiles.append(ref_product)
    
    
    # Canonicalizing target SMILES
    target_smiles_canonical = [
        _canonicalize_smiles(smi) for smi in target_smiles
    ]

    # Canonicalizing sampled SMILES
    sampled_smiles = [
        [_canonicalize_smiles(smi) for smi in smiles_list]
        for smiles_list in sampled_smiles
    ]
    
    top_Ks=np.array([1,3,5,10])
    acc,_ = _compute_accuracy(sampled_smiles, target_smiles_canonical, top_Ks=top_Ks)
    print(top_Ks)
    print(acc)

    res_df = {f"test_molecular_accuracy_top_{k}":acc[i] for i,k in enumerate(top_Ks)}
    
    if args.use_wandb:
        import wandb
        wandb.init(
            project="res_reaction",
            config = json.load(open(args.infer_config_filename))
        )
        wandb.config.update(args)
        wandb.log(res_df)
    