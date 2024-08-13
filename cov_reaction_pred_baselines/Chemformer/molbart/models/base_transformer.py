# from email.generator import Generator
import math
from functools import partial
import pdb
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from molbart.models.util import FuncLR

# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------- Base Transformer Model -----------------------------------------
# ----------------------------------------------------------------------------------------------------------


class _AbsTransformerModel(pl.LightningModule):
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
        schedule,
        warm_up_steps,
        dropout=0.1,
        num_beams=10,
        **kwargs,
    ):
        super().__init__()

        self.pad_token_idx = pad_token_idx
        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.max_seq_len = max_seq_len
        self.schedule = schedule
        self.warm_up_steps = warm_up_steps
        self.dropout = dropout

        if self.schedule == "transformer":
            assert (
                warm_up_steps is not None
            ), "A value for warm_up_steps is required for transformer LR schedule"

        # Additional args passed in to **kwargs in init will also be saved
        self.save_hyperparameters()

        # These must be set by subclasses
        self.sampler = None
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = num_beams
        self.n_unique_beams = num_beams

        self.emb = nn.Embedding(vocabulary_size, d_model, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

    def forward(self, x):
        raise NotImplementedError()

    def _calc_loss(self, batch_input, model_output):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor)
        """

        raise NotImplementedError()

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        self.train()

        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        with torch.no_grad():
            model_output = self.forward(batch)
            target_smiles = batch["target_smiles"]

            loss = self._calc_loss(batch, model_output)
            token_acc = self._calc_token_acc(batch, model_output)
            perplexity = self._calc_perplexity(batch, model_output)

            mol_strs, log_lhs = self.sample_molecules(
                batch, sampling_alg=self.val_sampling_alg
            )

            metrics = self.sampler.compute_sampling_metrics(mol_strs, target_smiles)

            mol_acc = torch.tensor(metrics["accuracy"], device=loss.device)
            invalid = torch.tensor(metrics["fraction_invalid"], device=loss.device)

            # Need to be logged for checkpointing callback
            self.log("mol_acc", mol_acc, prog_bar=True, logger=True, sync_dist=True)

            val_outputs = {
                "val_loss": loss,
                "val_token_accuracy": token_acc,
                "perplexity": perplexity,
                "val_molecular_accuracy": mol_acc,
                "val_invalid_smiles": invalid,
            }
        return val_outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)
        return

    def test_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            model_output = self.forward(batch)
            target_smiles = batch["target_smiles"]

            loss = self._calc_loss(batch, model_output)
            token_acc = self._calc_token_acc(batch, model_output)
            perplexity = self._calc_perplexity(batch, model_output)
            mol_strs, log_lhs = self.sample_molecules(
                batch, sampling_alg=self.test_sampling_alg
            )
            obj = {'reactants':batch['encoder_smiles'],'sampled_products': mol_strs, 'target_products': target_smiles}
            pickle.dump(obj, open('sampled_test.pkl', 'wb'))
            reactant_AA = [reactant.split(".")[0] for reactant in batch["encoder_smiles"]] # reactant = f"{AA}.{ligand}"
            metrics = self.sampler.compute_sampling_metrics(mol_strs, target_smiles)
            
            test_outputs = {
                "test_loss": loss.item(),
                "test_token_acc": token_acc.item(),
                "test_perplexity": perplexity.item(),
                # "target_smiles": target_smiles,
                # "sampled_molecules": mol_strs, # logger cannot output 2D tensor
                # "log_lhs": log_lhs, # logger cannot output 2D tensor
                # "test_invalid_smiles": metrics["invalid"], # there is no "invalid" key in metrics output
            }
            if hasattr(self.sampler, "sample_molecules"):
                test_outputs["test_molecular_accuracy"] = metrics["accuracy"]
                # test_outputs["top_Ks"] = metrics["top_Ks"]
                if metrics["fraction_invalid"]!=None:
                    test_outputs["fraction_invalid"] = metrics["fraction_invalid"]
                if metrics["fraction_unique"]!=None:
                    test_outputs["fraction_unique"] = metrics["fraction_unique"]
                if type(metrics['similarity'][0]) != float or np.isnan(metrics['similarity'][0]).any():
                    pass
                else:
                    test_outputs["similarity"] = metrics["similarity"]
                for k in [k for k in metrics.keys() if k.startswith('accuracy_top_')]: 
                    test_outputs.update({"test_molecular_"+k: metrics[k]})
            else:
                if self.test_sampling_alg == "greedy":
                    test_outputs["test_molecular_accuracy"] = metrics["accuracy"]

                elif self.test_sampling_alg == "beam":
                    test_outputs["test_molecular_accuracy"] = metrics["top_1_accuracy"]
                    test_outputs["test_molecular_top_1_accuracy"] = metrics[
                        "top_1_accuracy"
                    ]
                    test_outputs["test_molecular_top_2_accuracy"] = metrics[
                        "top_2_accuracy"
                    ]
                    test_outputs["test_molecular_top_3_accuracy"] = metrics[
                        "top_3_accuracy"
                    ]
                    test_outputs["test_molecular_top_5_accuracy"] = metrics[
                        "top_5_accuracy"
                    ]
                    test_outputs["test_molecular_top_10_accuracy"] = metrics[
                        "top_10_accuracy"
                    ]

                else:
                    raise ValueError(
                        f"Unknown test sampling algorithm, {self.test_sampling_alg}"
                    )

        for key, val in test_outputs.items():
            self.log(key, val, logger=True, sync_dist=True, on_step=True)
        return test_outputs

    def test_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999)
        )

        if self.schedule == "const":
            print("Using constant LR schedule.")
            const_sch = FuncLR(optim, lr_lambda=self._const_lr)
            sch = {"scheduler": const_sch, "interval": "step"}

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model**-0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step**-0.5, step * (self.warm_up_steps**-1.5))
        return self.lr * mult * lr

    def _const_lr(self, step):
        if self.warm_up_steps is not None and step < self.warm_up_steps:
            return (self.lr / self.warm_up_steps) * step

        return self.lr

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000**encs
        encs = [
            (torch.sin(pos / encs), torch.cos(pos / encs))
            for pos in range(self.max_seq_len)
        ]
        encs = [torch.stack(enc, dim=1).flatten()[: self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        """
        Method copied from Pytorch nn.Transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode
        """

        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        vocab_dist_output = model_output["token_output"]

        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = inv_target_mask.sum(dim=0)
        exp = -(1 / seq_lengths)
        # perp = torch.pow(log_probs.exp(), exp)
        perp = torch.exp(log_probs * exp)
        return perp.mean()

    def _calc_token_acc(self, batch_input, model_output):
        token_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def _avg_dicts(self, colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]

        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)
