
from typing import List, Tuple


import pdb
import argparse
import pandas as pd
import numpy as np
import tqdm

import rdkit
from rdkit import Chem

def statistics_compute_substracture_accuracy(
    sampled_smiles: List[List[str]], target_smiles: List[str], top_Ks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computing top-K accuracy for each K in 'top_Ks'.
    Modified version to return `is_in_set` for further analysis.
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
            pre_i_k = i_k -1 if i_k > 0 else 0
            pre_K = top_Ks[pre_i_k] if pre_i_k > 0 else 0
            for mol in mols[pre_K:K]:
                try:
                    if is_in_set[i_sample, pre_i_k] or is_in_set[i_sample, i_k]:
                        is_in_set[i_sample, i_k] = True
                        continue
                    mol = Chem.MolFromSmiles(mol)
                    target_mol = Chem.MolFromSmiles(target_smiles[i_sample])
                    if mol == None:
                        continue
                    is_in_set[i_sample, i_k] = mol.HasSubstructMatch(target_mol) and target_mol.HasSubstructMatch(mol)
                except:
                    pdb.set_trace()
    is_in_set = np.cumsum(is_in_set, axis=1)
    accuracy = np.mean(is_in_set > 0, axis=0)
    return accuracy, top_Ks, is_in_set

def statistics_compute_exact_match_accuracy(
    sampled_smiles: List[List[str]], target_smiles: List[str], top_Ks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computing top-K accuracy for each K in 'top_Ks'.
    Modified version to return `is_in_set` for further analysis.
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
    return accuracy, top_Ks, is_in_set

def statistics_duplicated(df):
    test_df = df[df['set']=='test'].dropna(subset=['reactants', 'products'])
    test_df['reactants_products'] = test_df.apply(lambda row: row['reactants'] + ">>" +row['products'], axis=1)
    reactants_dup = test_df.duplicated(subset=['reactants']).astype(int)
    reactants_products_dup = test_df.duplicated(subset=['reactants_products']).astype(int)
    dup_sum = reactants_dup + reactants_products_dup
    print('duplicated idx:')
    print(dup_sum[dup_sum==1])
    print(f'len of test set : {len(test_df)}, ratio: {len(dup_sum[dup_sum==1])/len(test_df)}')
    

if __name__=='__main__':
    import pickle
    obj = pickle.load(open('./cov_reaction_pred_baselines/Chemformer/sampled_test_viz_20240803_180052.pkl','rb'))
    reactants = obj['reactants']; sampled_smiles = obj['sampled_products']; target_smiles = obj['target_products']
    dataset_info_path = './data/processed/dataset.csv'
    save_path = './data/statistic/dataset.statistics.csv'
    
    # statistics_duplicated(pd.read_csv(dataset_info_path))
    
    top_Ks = np.array([1,3,5,10])
    substructure_accuracy, top_Ks, substructure_is_in_set = statistics_compute_substracture_accuracy(sampled_smiles, target_smiles, top_Ks)
    exact_match_accuracy, top_Ks, exact_match_is_in_set = statistics_compute_exact_match_accuracy(sampled_smiles, target_smiles, top_Ks)
    print(f'substructure_accuracy: {substructure_accuracy}')
    print(f'exact_match_accuracy: {exact_match_accuracy}')
    df = pd.read_csv(dataset_info_path, index_col=['reactants', 'products'])
    df = df[df['set']=='test'] # take test subset
    df['pred_products'] = None; df['substructure_res'] = None; df['exact_match_res'] = None
    for react, tar, pred, substructure_res, exact_match_res in zip(reactants,target_smiles, sampled_smiles, substructure_is_in_set, exact_match_is_in_set):
        df.loc[(react,tar), 'pred_products'] = pred[0]
        df.loc[(react,tar), 'substructure_res'] = substructure_res[0]
        df.loc[(react,tar), 'exact_match_res'] = exact_match_res[0]

    df.reset_index(drop=False, inplace=True)
    df.to_csv(save_path, index=False)