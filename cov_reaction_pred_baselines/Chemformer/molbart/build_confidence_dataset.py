import os
import pdb
import argparse
import random
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from rdkit import Chem

import molbart.modules.util as util
from molbart.models import Chemformer

DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_BEAMS = 10


def is_same_molecule(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 != None and mol2 != None:
        return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)
    return False

def write_predictions(args, reactants,smiles, log_lhs, gt_product, split):
    pickle.dump([reactants, smiles, log_lhs, gt_product, split], open('./tmp/predictions.pkl', 'wb'))
    confidence_data = []
    for r, ps, gt, s in zip(reactants, smiles, gt_product, split):
        ps = ps.tolist()
        pos = None; negs = []
        for p in ps:
            if not is_same_molecule(p,gt):
                negs.append(p)
                if len(negs)>=1:
                    break # TODO
        pos = gt
        if len(negs) == 0:
            continue # TODO
        
        # positive sample
        confidence_data.append({
            "rxn": r + '>>' + pos,
            # "confidence": random.uniform(0.8, 1),
            "confidence": 1,
            "set": s
        })
        # negative sample
        for neg in negs:
            confidence_data.append({
                "rxn": r + '>>' + neg,
                # "confidence": random.uniform(0, 0.2),
                "confidence": 0,
                "set": s
            })
    confidence_df = pd.DataFrame(confidence_data)
    confidence_df.to_csv(args.save_path, index=False)
    

def main(args):
    os.makedirs('./tmp', exist_ok=True)
    args.reactants_path = './tmp/' + 'rxn.txt'
    df = pd.read_csv(args.data_path).dropna(subset=['reactants', 'products'])
    # df = df[df['set'] != 'test'].reset_index()
    df['rxn'] = df.apply(lambda x: x['reactants'] + '>>' + x['products'], axis=1)
    pd.set_option("display.max_colwidth", 100000)
    rxn = df['rxn'].to_string(index=False)
    open(args.reactants_path, 'wt').write(rxn)
    model_args, data_args = util.get_chemformer_args(args)

    kwargs = {
        "vocabulary_path": args.vocabulary_path,
        "n_gpus": args.n_gpus,
        "model_path": args.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "n_beams": args.n_beams,
        "train_mode": "eval",
    }
    chemformer = Chemformer(
        **kwargs,
        datamodule_type="simple_reaction_list", 
    )

    print("Making predictions...")
    smiles, log_lhs, gt_product = chemformer.predict(dataset=args.dataset_part)
    reactants = df['reactants'].tolist(); split = df["set"].tolist()
    # reactants, smiles, log_lhs, gt_product, split = pickle.load(open('./tmp/predictions.pkl', 'rb'))
    write_predictions(args, reactants,smiles, log_lhs, gt_product, split)
    print("Finished predictions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="The path and filename of the data file to load (will use the train and valid sets only)")
    parser.add_argument("--save_path", help="The path and filename of the data file to save")

    parser.add_argument(
        "--dataset_part",
        help="Specifies which part of dataset to use.",
        choices=["full", "train", "val", "test"],
        default="full",
    )
    parser.add_argument("--vocabulary_path", default=util.DEFAULT_VOCAB_PATH)

    parser.add_argument(
        "--task",
        choices=["forward_prediction", "backward_prediction", "mol_opt"],
        default="forward_prediction",
    )

    # Model args
    parser.add_argument(
        "--model_type", choices=["bart", "unified"], default=util.DEFAULT_MODEL
    )
    parser.add_argument("--model_path")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n_beams", type=int, default=DEFAULT_NUM_BEAMS)

    parser.add_argument("--n_gpus", type=int, default=util.DEFAULT_GPUS)

    args = parser.parse_args()
    
    main(args)