from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from molbart.models import _AbsTransformerModel
from molbart.models.util import PreNormEncoderLayer

import os
import pdb
from argparse import Namespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import scipy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import molbart.modules.util as util
from molbart.models import BARTModel, UnifiedModel
from molbart.modules.data.base import SimpleReactionListDataModule
from molbart.modules.decoder import BeamSearchSampler
from molbart.modules.tokenizer import ChemformerTokenizer

DEFAULT_WEIGHT_DECAY = 0


import logging
logging.getLogger('pysmilesutils').setLevel(logging.ERROR)
# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Pre-train Models --------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class BERTModel(_AbsTransformerModel):
    def __init__(
        self,
        pad_token_idx,
        vocabulary_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        temperature=0.05,
        **kwargs,
    ):
        super().__init__(
            pad_token_idx,
            vocabulary_size,
            d_model,
            num_layers,
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs,
        )

        self.encoder = nn.TransformerEncoder(
            PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # self.pool_fc = nn.Linear(d_model, 1)
        self.pool_fc = partial(torch.sum, dim=0)
        self.temperature = temperature
        # self.loss_function = nn.BCEWithLogitsLoss(reduce='none')
        
        # self.ffn = nn.Sequential(
        #     nn.Linear(d_model,d_model),
        #     nn.GELU(),
        #     nn.Dropout(p=dropout),
        # )
        # self.out = nn.Linear(d_model,1)
        
        
        self._init_params()

    def forward(self, x):
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """
        reactants_emb = self.encode(x['reactants_input'], x['reactants_pad_mask'])
        products_emb = self.encode(x['products_input'], x['products_pad_mask'])
        
        # reactants_emb = self.pool_fc(reactants_emb)
        # products_emb = self.pool_fc(products_emb)
        reactants_emb = reactants_emb[0]
        products_emb = products_emb[0]
        # token_output = self.pool_fc(embeddings[0]).squeeze() # forward on bos token
        # output = {"model_output": embeddings, "token_output": token_output}
        output = {
            "reactants_emb": reactants_emb,
            "products_emb": products_emb
        }

        return output

    def encode(self, encoder_input, encoder_pad_mask):
        """Construct the embedding for an encoder input

        Args:
            "encoder_input": tensor of token_ids of shape (src_len, batch_size),
            "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),

        Returns:
            encoder embedding (Tensor of shape (seq_len, batch_size, d_model))
        """
        encoder_pad_mask = encoder_pad_mask.transpose(0,1)
        encoder_embs = self._construct_input(encoder_input) # (src_len, batch_size, d_model)
        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def _calc_loss(self, batch_input, model_output):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """
        
        target = batch_input["target"]
        reactants_emb = model_output['reactants_emb']
        products_emb = model_output['products_emb']
        
        def calculate_loss(reactants_emb, products_emb, target, temperature):
            assert target.any(), 'There should be at least a postive example in a batch, but got all negtive examples. Maybe try a larger batch size'
            pos_idx = torch.where(target == 1)[0]
            neg_idx = torch.where(target == 0)[0]
            r_emb = reactants_emb[pos_idx]
            pos_emb = products_emb[pos_idx]
            neg_emb = products_emb[neg_idx]

            pos_cos_sim = torch.nn.functional.cosine_similarity(r_emb[:,None,:], pos_emb[None,:,:], dim=-1) / temperature # (r, pos_num), pos_num should equal to r
            neg_cos_sim = torch.nn.functional.cosine_similarity(r_emb[:,None,:], neg_emb[None,:,:], dim=-1) / temperature # (r, neg_num)
            cos_sim = torch.cat([pos_cos_sim, neg_cos_sim], dim=1)
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            loss = torch.nn.functional.cross_entropy(cos_sim, labels)
            return loss

        loss = calculate_loss(reactants_emb, products_emb, target, self.temperature)
        return loss

    def _calc_acc(self, batch, model_output):
        target = batch['target']
        dist = self._calc_dist(model_output["reactants_emb"] , model_output["products_emb"] )
        pred = torch.where(dist <= 5, 1, 0)
        return torch.mean((pred == target).float())
    
    @staticmethod
    def _calc_dist(reactants_emb, products_emb):
        return nn.functional.pairwise_distance(reactants_emb, products_emb, p=2)

    def validation_step(self, batch, batch_idx):
        self.eval()

        with torch.no_grad():
            model_output = self.forward(batch)

            loss = self._calc_loss(batch, model_output)

            acc = self._calc_acc(batch, model_output)

            # Need to be logged for checkpointing callback
            self.log("acc", acc, prog_bar=True, logger=True, sync_dist=True)

            val_outputs = {
                "val_loss": loss,
                "val_token_accuracy": acc,
                
                "perplexity": 0,
                "val_molecular_accuracy": 0,
                "val_invalid_smiles": 0,
            }
        return val_outputs

    def test_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            model_output = self.forward(batch)

            loss = self._calc_loss(batch, model_output)
            acc = self._calc_acc(batch, model_output)
            
            test_outputs = {
                "test_loss": loss.item(),
                "test_token_acc": acc,
                
                "test_perplexity": 0,
            }
            
        for key, val in test_outputs.items():
            self.log(key, val, logger=True, sync_dist=True, on_step=True)
        return test_outputs

class ConfidenceModel:
    """
    Class for building (synthesis) Chemformer model, fine-tuning seq-seq model,
    and predicting/scoring model.
    """

    def __init__(
        self,
        vocabulary_path: str,
        model_args: Namespace,
        data_args: Namespace,
        model_path: Optional[str] = None,
        n_gpus: int = 1,
        datamodule_type: str = "seq2seq",
        train_mode: str = "training",
        device: str = "cuda",
        data_device: str = "cuda",
        build_trainer: bool = False,
        resume_training: bool = False,
    ) -> None:
        """
        Args:
            vocabulary_path (str): path to bart_vocabulary.
            model_args (Namespace): Arguments for building the chemformer model.
            data_args (Namespace): Arguments for building torch datamodule.
            model_path (Optional[str]): Path to model weights.
            n_gpus (int): Number of GPUs to use.
            n_beams (int): Number of beams in beam search.
            n_unique_beams (Optional[int]): Restrict number of unique beam search solutions.
                If None => return all unique solutions.
            datamodule_type (str): The type of datamodule to build.
            train_model (str): Whether to train the model ("training") or use
                model for evaluations ("eval").
            sampler (str): Which beam search sampler to use ("optimized" => GPU
                optimized beam search).
            device (str): Which device to run model and beam search on ("cuda" / "cpu").
            data_device (str): device used for handling the data in optimized beam search.
                If memory issues, could help to set data_device="cpu"
            build_trainer (bool): If True, build a trainer which can be used for
                fine-tuning the model.
            sample_unique (bool): Whether to return unique beam search solutions from the
                optimized beam search.
            resume_training (bool): Whether to continue training from the supplied
                .ckpt file.
        """

        self.train_mode = train_mode
        self.resume_training = resume_training
        if resume_training:
            print("Resuming training.")

        if n_gpus < 1:
            device = "cpu"
            data_device = "cpu"

        self.device = device

        self.tokenizer = ChemformerTokenizer(filename=vocabulary_path)
        self.train_tokens = data_args.train_tokens
        self.n_buckets = data_args.n_buckets

        self.model_type = model_args.model_type
        self.model_path = model_path
        if self.model_path is None:
            self.model_path = "None"

        self.data_args = data_args
        self.n_gpus = n_gpus
        self.is_data_setup = False
        self.set_datamodule(datamodule_type=datamodule_type)

        print("Vocabulary_size: " + str(len(self.tokenizer)))
        self.vocabulary_size = len(self.tokenizer)

        if self.train_mode == "training" or self.train_mode == "train":
            self.train_steps = util.calc_train_steps(
                model_args, self.datamodule, n_gpus
            )
            print(f"Train steps: {self.train_steps}")

        print("Building model.")
        self.build_model(model_args)

        if (
            self.train_mode == "training" or self.train_mode == "train"
        ) or build_trainer:
            print("Building trainer.")
            self.trainer = util.build_trainer(
                model_args, n_gpus, data_args.dataset_type
            )
            if not self.resume_training:
                self.out_directory = self._set_out_directory()
            print("Model initialization done.")

        self.model.to(device)
        return

    def encode(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
    ) -> List[torch.Tensor]:
        """
        Compute memory from transformer inputs.

        Args:
            dataset (str): (Which part of the dataset to use (["train", "val", "test",
                "full"]).)
            dataloader (DataLoader): (If None -> dataloader
                will be retrieved from self.datamodule)
        Returns:
            List[torch.Tensor]: Tranformer memory
        """

        self.model.to(self.device)
        self.model.eval()

        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        X_encoded = []
        for b_idx, batch in enumerate(dataloader):
            batch = self.on_device(batch)
            with torch.no_grad():
                batch_encoded = self.model.encode(batch).permute(
                    1, 0, 2
                )  # Return on shape [n_samples, n_tokens, max_seq_length]

            X_encoded.extend(batch_encoded)
        return X_encoded

    def _set_out_directory(self) -> str:
        """
        Defining the output directory for fine-tuned model weights. Will create a new
        version_[X] directory to not overwrite previous training sessions.

        Returns:
            str: Output directory
        """
        if self.trainer.logger is not None:
            if self.trainer.weights_save_path != self.trainer.default_root_dir:
                save_dir = self.trainer.weights_save_path
            else:
                save_dir = self.trainer.logger.save_dir or self.trainer.default_root_dir

            version = (
                self.trainer.logger.version
                if isinstance(self.trainer.logger.version, str)
                else f"version_{self.trainer.logger.version}"
            )
            version, name = self.trainer.training_type_plugin.broadcast(
                (version, self.trainer.logger.name)
            )
            out_dir = os.path.join(save_dir, str(name), version)
        else:
            out_dir = self.trainer.weights_save_path
        return out_dir

    def set_datamodule(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        datamodule_type: Optional[str] = None,
    ) -> None:
        """
        Create a new datamodule by either supplying a datamodule (created elsewhere) or
        a pre-defined datamodule type as input.

        Args:
            datamodule (Optional[pl.LightningDataModule]): pytorchlightning datamodule
            datamodule_type (Optional[str]): The type of datamodule to build if no
                datamodule is given as input.
        """
        if datamodule is None and datamodule_type is not None:
            print("Datamodule type: " + str(datamodule_type))
            if datamodule_type == "seq2seq":
                self.datamodule = util.build_seq2seq_datamodule(
                    self.data_args, self.tokenizer, self.data_args.forward_prediction
                )
            else:
                raise ValueError(f"Unknown datamodule type: {datamodule_type}")
        elif datamodule is None:
            print("Did not initialize datamodule.")
            return
        else:
            self.datamodule = datamodule

        self.datamodule.setup()
        n_cpus = len(os.sched_getaffinity(0))
        if self.n_gpus > 0:
            n_workers = n_cpus // self.n_gpus
        else:
            n_workers = n_cpus
        self.datamodule._num_workers = n_workers
        print(f"Using {str(n_workers)} workers for data module.")
        return

    def fit(self) -> None:
        """
        Fit model to training data in self.datamodule and using parameters specified in
        the trainer object.
        """
        self.trainer.fit(self.model, datamodule=self.datamodule)
        return

    def parameters(self) -> Iterator:
        return self.model.parameters()

    def _random_initialization(
        self, args: Namespace, extra_args: Dict[str, Any], pad_token_idx: int
    ) -> Union[BARTModel, UnifiedModel]:
        """
        Constructing a model with randomly initialized weights.

        Args:
            args (Namespace): Grouped model arguments.
            extra_args (Dict[str, Any]): Extra arguments passed to the BARTModel.
            Will be saved as hparams by pytorchlightning.
            pad_token_idx: The index denoting padding in the vocabulary.
        """

        total_steps = self.train_steps + 1

        if self.model_type == "bert":
            model = BERTModel(
                pad_token_idx,
                self.vocabulary_size,
                args.d_model,
                args.n_layers,
                args.n_heads,
                args.d_feedforward,
                args.learning_rate,
                DEFAULT_WEIGHT_DECAY,
                util.DEFAULT_ACTIVATION,
                total_steps,
                util.DEFAULT_MAX_SEQ_LEN,
                schedule=args.schedule,
                dropout=args.dropout,
                warm_up_steps=args.warm_up_steps,
                **extra_args,
            )
        else:
            raise ValueError(f"Unknown model type [bert]: {self.model_type}")

        return model

    def _initialize_from_ckpt(
        self, args: Namespace, extra_args: Dict[str, Any], pad_token_idx: int
    ) -> Union[BARTModel, UnifiedModel]:
        """
        Constructing a model with weights from a ckpt-file.

        Args:
            args (Namespace): Grouped model arguments.
            extra_args (Dict[str, Any]): Extra arguments passed to the BARTModel.
            Will be saved as hparams by pytorchlightning.
            pad_token_idx: The index denoting padding in the vocabulary.
        """
        if self.train_mode == "training" or self.train_mode == "train":
            total_steps = self.train_steps + 1

        if self.model_type == "bert":
            if self.train_mode == "training" or self.train_mode == "train":
                if self.resume_training:
                    model = BERTModel.load_from_checkpoint(
                        self.model_path,
                        num_steps=total_steps,
                        pad_token_idx=pad_token_idx,
                        vocabulary_size=self.vocabulary_size,
                    )
                else:
                    model = BERTModel.load_from_checkpoint(
                        self.model_path,
                        pad_token_idx=pad_token_idx,
                        vocabulary_size=self.vocabulary_size,
                        num_steps=total_steps,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        schedule=args.schedule,
                        warm_up_steps=args.warm_up_steps,
                        dropout = args.dropout,
                        strict = False,
                        **extra_args,
                    )
            elif (
                self.train_mode == "validation"
                or self.train_mode == "val"
                or self.train_mode == "test"
                or self.train_mode == "testing"
                or self.train_mode == "eval"
            ):
                model = BERTModel.load_from_checkpoint(
                    self.model_path
                )
                model.eval()
            else:
                raise ValueError(f"Unknown training mode: {self.train_mode}")
        else:
            raise ValueError(f"Unknown model type [bert]: {self.model_type}")
        return model

    def build_model(self, args: Namespace) -> None:
        """
        Build transformer model, either
        1. By loading pre-trained model from checkpoint file, or
        2. Initializing new model with random weight initialization

        Args:
            args (Namespace): Grouped model arguments.
        """

        pad_token_idx = self.tokenizer["pad"]

        # These args don't affect the model directly but will be saved by lightning as hparams
        # Tensorboard doesn't like None so we need to convert to string
        train_tokens = "None" if self.train_tokens is None else self.train_tokens
        n_buckets = "None" if self.n_buckets is None else self.n_buckets

        if self.train_mode == "training" or self.train_mode == "train":
            extra_args = {
                "batch_size": self.datamodule.batch_size,
                "acc_batches": args.acc_batches,
                "epochs": args.n_epochs,
                "clip_grad": args.clip_grad,
                "augment": args.augmentation_strategy,
                "aug_prob": args.augmentation_probability,
                "train_tokens": train_tokens,
                "n_buckets": n_buckets,
                "limit_val_batches": args.limit_val_batches,
                "contrastive_loss_margin": args.contrastive_loss_margin
            }
        else:
            extra_args = {}

        # If no model is given, use random init
        if self.model_path in ["none", "None"]:
            self.model = self._random_initialization(args, extra_args, pad_token_idx)
        else:
            self.model = self._initialize_from_ckpt(args, extra_args, pad_token_idx)
        return

    def get_logged_data(self) -> pd.DataFrame:
        """
        Build dataframe from logged metrics, which are stored in self.trainer.callbacks[2]

        Returns:
            pd.DataFrame: DataFrame with training-loss, validation-loss and molecular
            accuracy of the validation set.
        """
        epochs = self.trainer.callbacks[1].epochs
        train_loss = self.trainer.callbacks[1].train_loss
        val_loss = self.trainer.callbacks[1].val_loss
        val_acc = self.trainer.callbacks[1].val_token_acc

        epochs = np.array(epochs).ravel()
        train_loss = np.array(
            [sample.to(torch.device("cpu")).numpy() for sample in train_loss]
        ).ravel()
        val_loss = np.array(
            [sample.to(torch.device("cpu")).numpy() for sample in val_loss]
        ).ravel()
        val_acc = np.array(
            [sample.to(torch.device("cpu")).numpy() for sample in val_acc]
        ).ravel()

        df = pd.DataFrame(epochs, columns=["epoch"])
        df["training_loss"] = train_loss
        df["validation_loss"] = val_loss
        df["validation_token_accuracy"] = val_acc
        
        return df

    def save_logged_data(self) -> None:
        """
        Retrieve and write data (model validation) logged during training.
        """
        metrics_df = self.get_logged_data()
        if self.resume_training:
            self._set_out_directory()
        outfile = self.out_directory + "/logged_train_metrics.csv"
        metrics_df.to_csv(outfile, sep="\t", index=False)
        print("Logged training/validation set loss written to: " + outfile)
        return

    def get_dataloader(
        self, dataset: str, datamodule: Optional[pl.LightningDataModule] = None
    ) -> DataLoader:
        """
        Get the dataloader for a subset of the data from a specific datamodule.

        Args:
            dataset (str): One in ["full", "train", "val", "test"].
                Specifies which part of the data to return.
            datamodule (Optional[pl.LightningDataModule]): pytorchlightning datamodule.
                If None -> Will use self.datamodule.
        """
        if dataset not in ["full", "train", "val", "test"]:
            raise ValueError(
                f"Unknown dataset : {dataset}. Should be either 'full', 'train', 'val' or 'test'."
            )

        if datamodule is None:
            datamodule = self.datamodule

        dataloader = None
        if dataset == "full":
            dataloader = datamodule.full_dataloader()
        elif dataset == "train":
            dataloader = datamodule.train_dataloader()
        elif dataset == "val":
            dataloader = datamodule.val_dataloader()
        elif dataset == "test":
            dataloader = datamodule.test_dataloader()

        return dataloader

    def on_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move data in "batch" to the current model device.

        Args:
            batch (Dict[str, Any]): batch input data to model.
        Returns:
            Dict[str, Any]: batch data on current device.
        """
        device_batch = {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()
        }
        return device_batch

    def predict(
        self,
        dataset: str = "full",
        dataloader: Optional[DataLoader] = None,
        return_tokenized: bool = False,
        i_chunk: int = 0,
        n_chunks: int = 1,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Predict embedding distance output given dataloader, specified by 'dataset'.
        Args:
            dataset (str): Which part of the dataset to use (["train", "val", "test",
                "full"]).
            dataloader (Optional[DataLoader]): If None -> dataloader
                will be retrieved from self.datamodule.
            return_tokenized (bool): Whether to return the tokenized beam search
                solutions instead of strings.
        Returns:
            (sampled_smiles List[np.ndarray], log_lhs List[np.ndarray], target_smiles List[np.ndarray])
        """
        if dataloader is None:
            dataloader = self.get_dataloader(dataset)

        # Divide batches into chunks of batches
        n_batches_in_chunk = int(len(dataloader) / float(n_chunks))
        start_batch_idx = i_chunk * n_batches_in_chunk

        self.model.to(self.device)
        self.model.eval()

        preds = []
        target = []
        input_smiles = []
        for b_idx, batch in enumerate(dataloader):
            if n_chunks > 1:
                if b_idx < start_batch_idx:
                    continue

                if i_chunk != n_chunks - 1:
                    if (
                        i_chunk != n_chunks - 1
                        and b_idx == start_batch_idx + n_batches_in_chunk
                    ):
                        break

            batch = self.on_device(batch)
            with torch.no_grad():
                output = self.model.forward(batch)

            target.extend(batch["target"].cpu().tolist())
            preds.extend(self.model._calc_dist(output["reactants_emb"] , output["products_emb"] ).cpu().tolist())
            input_smiles.extend([f"{r}>>{p}" for r,p in zip(batch['reactants_smiles'], batch['products_smiles'])])

        return preds, target, input_smiles