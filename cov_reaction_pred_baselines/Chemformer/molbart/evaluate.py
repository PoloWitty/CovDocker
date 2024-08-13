import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import molbart.modules.util as util
from molbart.models import Chemformer

DEFAULT_BATCH_SIZE = 64
DEFAULT_AUG_PROB = 0.0
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = None
DEFAULT_NUM_BEAMS = 10
DEFAULT_VOCAB_PATH = "bart_vocab_downstream.txt"


def build_trainer(args, limit_test_batches=1.0):
    logger = TensorBoardLogger(
        "tb_logs", name=f"eval_{args.model_type}_{args.dataset_type}"
    )
    trainer = Trainer(
        logger=logger, limit_test_batches=limit_test_batches, precision=16, gpus=1
    )
    return trainer


def main(args):
    util.seed_everything(args.seed)

    if args.dataset_type not in [
        "uspto_mixed",
        "uspto_50",
        "uspto_sep",
        "uspto_50_with_type",
        "synthesis",
        "covdocker_synthesis"
    ]:
        raise ValueError(f"Unknown dataset: {args.dataset_type}")

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

    chemformer = Chemformer(**kwargs)

    trainer = build_trainer(args)
    print("Evaluating model...")
    results = trainer.test(chemformer.model, datamodule=chemformer.datamodule)

    print(f"Results for model: {args.model_path}")
    print(f"{'Item':<25}Result")
    for key, val in results[0].items():
        print(f"{key:<25} {val:.4f}")
    print("Finished evaluation.")
    
    if args.use_wandb:
        import wandb
        wandb.init(
            project="res_reaction",
            config = kwargs
        )
        wandb.config.update(args)
        wandb.log(results[0])
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path")
    parser.add_argument("--model_path")
    parser.add_argument("--dataset_type", default=util.DEFAULT_DATASET_TYPE)
    parser.add_argument(
        "--model_type", choices=["bart", "unified"], default=util.DEFAULT_MODEL
    )
    parser.add_argument("--task", choices=["forward_prediction", "backward_prediction"])
    parser.add_argument("--vocabulary_path", default=DEFAULT_VOCAB_PATH)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=util.DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--n_beams", type=int, default=DEFAULT_NUM_BEAMS)
    parser.add_argument("--n_gpus", type=int, default=util.DEFAULT_GPUS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--n_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument(
        "-aug_prob", "--augmentation_probability", type=float, default=DEFAULT_AUG_PROB
    )
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--use-wandb",type=int, default=0, help='whether use wandb to store result')
    parser.add_argument("--run-id", type=str, default='default-run-id')
    
    
    args = parser.parse_args()
    main(args)
