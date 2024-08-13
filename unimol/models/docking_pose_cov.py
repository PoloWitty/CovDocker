# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from .unimol import UniMolModel, base_architecture, NonLinearHead, GaussianLayer
from unicore.modules import LayerNorm
from .transformer_encoder_with_pair import TransformerEncoderWithPair
import numpy as np
import random

logger = logging.getLogger(__name__)


@register_model("docking_pose_cov")
class DockingPoseCovModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )
        parser.add_argument(
            "--refine",
            type=int,
            default=1,
            help="iter nums of refine",
        )

    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        unimol_docking_architecture(args)

        self.args = args
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)
        self.concat_decoder = TransformerEncoderWithPair(
            encoder_layers=4,
            embed_dim=args.mol.encoder_embed_dim,
            ffn_embed_dim=args.mol.encoder_ffn_embed_dim,
            attention_heads=args.mol.encoder_attention_heads,
            emb_dropout=0.1,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            activation_fn="gelu",
        )
        self.cross_distance_project = NonLinearHead(
            args.mol.encoder_embed_dim * 2 + args.mol.encoder_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, "relu"
        )
        
        self.mol_dict = mol_dictionary
        self.pocket_dict = pocket_dictionary
        if self.args.refine > 1: 
            # cross edge proj
            K = 128
            self.cross_edge_type = len(mol_dictionary) * len(pocket_dictionary)
            self.cross_gbf_proj = NonLinearHead(
                K, args.encoder_attention_heads, args.activation_fn
            )
            self.cross_gbf = GaussianLayer(K, self.cross_edge_type)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)

    def forward_once(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        cross_dist_in,
        masked_tokens=None,
        features_only=True,
        **kwargs
    ):
        def get_cross_edge_type(mol_tokens, pocket_tokens):
            bsz = mol_tokens.size(0)
            et = mol_tokens.view(bsz,-1,1) * len(self.pocket_dict) + pocket_tokens.view(bsz,1,-1)
            return et
        def get_dist_features(dist, et, flag):
            if flag == "mol":
                n_node = dist.size(-1)
                gbf_feature = self.mol_model.gbf(dist, et)
                gbf_result = self.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == "pocket":
                n_node = dist.size(-1)
                gbf_feature = self.pocket_model.gbf(dist, et)
                gbf_result = self.pocket_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == "cross":
                n_mol, n_pocket = dist.size(-2), dist.size(-1)
                gbf_feature = self.cross_gbf(dist, et)
                gbf_result = self.cross_gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result # (b, n_l, n_p, d)
                # graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                # graph_attn_bias = graph_attn_bias.view(-1, n_mol, n_pocket)
                return graph_attn_bias
            else:
                raise NotImplementedError
            
        # get ligand's token emb and pair emb
        mol_padding_mask = mol_src_tokens.eq(self.mol_model.padding_idx)
        mol_x = self.mol_model.embed_tokens(mol_src_tokens)
        mol_graph_attn_bias = get_dist_features(
            mol_src_distance, mol_src_edge_type, "mol"
        )
        mol_outputs = self.mol_model.encoder(
            mol_x, padding_mask=mol_padding_mask, attn_mask=mol_graph_attn_bias
        )
        mol_encoder_rep = mol_outputs[0]
        mol_encoder_pair_rep = mol_outputs[1]
        
        # get pocket's token emb and pair emb
        pocket_padding_mask = pocket_src_tokens.eq(self.pocket_model.padding_idx)
        pocket_x = self.pocket_model.embed_tokens(pocket_src_tokens)
        pocket_graph_attn_bias = get_dist_features(
            pocket_src_distance, pocket_src_edge_type, "pocket"
        )
        pocket_outputs = self.pocket_model.encoder(
            pocket_x, padding_mask=pocket_padding_mask, attn_mask=pocket_graph_attn_bias
        )
        pocket_encoder_rep = pocket_outputs[0]
        pocket_encoder_pair_rep = pocket_outputs[1]


        mol_sz = mol_encoder_rep.size(1)
        pocket_sz = pocket_encoder_rep.size(1)

        # concat ligand pocket features
        concat_rep = torch.cat(
            [mol_encoder_rep, pocket_encoder_rep], dim=-2
        )  # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat(
            [mol_padding_mask, pocket_padding_mask], dim=-1
        )  # [batch, mol_sz+pocket_sz]
        attn_bs = mol_graph_attn_bias.size(0)
        

        concat_attn_bias = torch.zeros(
            attn_bs, mol_sz + pocket_sz, mol_sz + pocket_sz
        ).type_as(
            concat_rep
        )  # [batch, mol_sz+pocket_sz, mol_sz+pocket_sz]
        concat_attn_bias[:, :mol_sz, :mol_sz] = (
            mol_encoder_pair_rep.permute(0, 3, 1, 2)
            .reshape(-1, mol_sz, mol_sz)
            .contiguous()
        )
        concat_attn_bias[:, -pocket_sz:, -pocket_sz:] = (
            pocket_encoder_pair_rep.permute(0, 3, 1, 2)
            .reshape(-1, pocket_sz, pocket_sz)
            .contiguous()
        )
        
        if self.args.refine > 1:
            # cross attn bias from cross_dist_in
            cross_et = get_cross_edge_type(mol_src_tokens, pocket_src_tokens)
            cross_graph_attn_bias = get_dist_features(
                cross_dist_in, cross_et, "cross"
            ) # (b, n_l, n_p, d)
            concat_attn_bias[:, :mol_sz, -pocket_sz:] = (
                cross_graph_attn_bias.permute(0, 3, 1, 2)
                .reshape(-1, mol_sz, pocket_sz)
                .contiguous()
            )
            concat_attn_bias[:, -pocket_sz:, :mol_sz] = (
                cross_graph_attn_bias.permute(0, 3, 2, 1)
                .reshape(-1, pocket_sz, mol_sz)
                .contiguous()
            )

        decoder_rep = concat_rep
        decoder_pair_rep = concat_attn_bias
        for i in range(self.args.recycling):
            decoder_outputs = self.concat_decoder(
                decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep
            )
            decoder_rep = decoder_outputs[0]
            decoder_pair_rep = decoder_outputs[1]
            if i != (self.args.recycling - 1):
                decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(
                    -1, mol_sz + pocket_sz, mol_sz + pocket_sz
                )

        mol_decoder = decoder_rep[:, :mol_sz]
        pocket_decoder = decoder_rep[:, mol_sz:]

        mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
        mol_pocket_pair_decoder_rep = (
            decoder_pair_rep[:, :mol_sz, mol_sz:, :]
            + decoder_pair_rep[:, mol_sz:, :mol_sz, :].transpose(1, 2)
        ) / 2.0
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

        cross_rep = torch.cat(
            [
                mol_pocket_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

        cross_distance_predict = (
            F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )  # batch, mol_sz, pocket_sz

        holo_encoder_pair_rep = torch.cat(
            [
                mol_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, mol_sz, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, mol_sz, 3*hidden_size]
        holo_distance_predict = self.holo_distance_project(
            holo_encoder_pair_rep
        )  # batch, mol_sz, mol_sz

        return cross_distance_predict, holo_distance_predict

    def forward(
        self,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        masked_tokens=None,
        features_only=True,
        **kwargs
    ):
        # init output
        bsz, mol_sz = mol_src_tokens.shape
        bsz, pocket_sz = pocket_src_tokens.shape
        cross_dist = torch.zeros((bsz, mol_sz, pocket_sz), dtype=mol_src_distance.dtype, device=mol_src_distance.device)
        holo_dist = mol_src_distance
        
        # Sample a random integer from a uniform distribution if training
        refine_num = random.randint(1, self.args.refine) if torch.is_grad_enabled() else self.args.refine
        # stop gradient for recycling -1 times
        with torch.no_grad():
            for i in range(refine_num-1):
                cross_dist, holo_dist = self.forward_once(mol_src_tokens, holo_dist, mol_src_edge_type, pocket_src_tokens, pocket_src_distance, pocket_src_edge_type, cross_dist, masked_tokens, features_only, **kwargs)

        cross_dist, holo_dist = self.forward_once(mol_src_tokens, holo_dist, mol_src_edge_type, pocket_src_tokens, pocket_src_distance, pocket_src_edge_type, cross_dist, masked_tokens, features_only, **kwargs)
        return cross_dist, holo_dist


    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float("-inf")] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@register_model_architecture("docking_pose_cov", "docking_pose_cov")
def unimol_docking_architecture(args):

    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(
        args, "pocket_encoder_ffn_embed_dim", 2048
    )
    args.pocket.encoder_attention_heads = getattr(
        args, "pocket_encoder_attention_heads", 64
    )
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(
        args, "pocket_pooler_activation_fn", "tanh"
    )
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0

    base_architecture(args)
