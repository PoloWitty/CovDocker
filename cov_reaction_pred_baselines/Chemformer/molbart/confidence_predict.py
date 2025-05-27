import argparse
from pathlib import Path
import pdb

import pandas as pd

import molbart.modules.util as util
from molbart.models import ConfidenceModel

DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_BEAMS = 10
DEFAULT_MODEL_TYPE = "bert"


def write_predictions(args, pred_dist, target, input_smiles):
    num_data = len(target)

    df_data = {
        "rxn": input_smiles,
        "pred_dist": pred_dist,
        "target": target
    }
    df = pd.DataFrame(data=df_data)
    df.to_csv(Path(args.dist_output_path),index=False)


def main(args):
    model_args, data_args = util.get_chemformer_args(args)

    kwargs = {
        "vocabulary_path": args.vocabulary_path,
        "n_gpus": args.n_gpus,
        "model_path": args.model_path,
        "model_args": model_args,
        "data_args": data_args,
        "train_mode": "eval",
    }

    confidence_model = ConfidenceModel(**kwargs)

    print("Making predictions...")
    pred_dist, target, input_smiles = confidence_model.predict(dataset=args.dataset_part)
    write_predictions(args, pred_dist, target, input_smiles)
    print("Finished predictions.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path")
    parser.add_argument("--model_path")
    parser.add_argument("--dist_output_path")
    parser.add_argument(
        "--dataset_part",
        help="Specifies which part of dataset to use.",
        choices=["full", "train", "val", "test"],
        default="full",
    )
    parser.add_argument(
        "--dataset_type",
        help="The datamodule to to use, example 'uspto_50' (see molbart.util.build_seq2seq_datamodule",
    )
    parser.add_argument("--vocabulary_path", default=util.DEFAULT_VOCAB_PATH)

    parser.add_argument(
        "--task",
        choices=["forward_prediction", "backward_prediction", "mol_opt"],
        default="forward_prediction",
    )

    # Model args
    parser.add_argument(
        "--model_type", choices=["bert"], default=DEFAULT_MODEL_TYPE
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n_beams", type=int, default=DEFAULT_NUM_BEAMS)

    parser.add_argument("--n_gpus", type=int, default=util.DEFAULT_GPUS)

    args = parser.parse_args()
    main(args)
