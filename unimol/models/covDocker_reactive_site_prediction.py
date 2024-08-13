"""
desc:	reactive site prediction model
author:	Yangzhe Peng
date:	2024/05/12
"""


import pdb
import logging
from typing import Optional, Dict, Any, List
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from unicore.modules import TransformerEncoderLayer
from .unimol import UniMolModel
from unicore.modules import LayerNorm
from .unimol import base_architecture
from .cross_attention import CrossAttention

logger = logging.getLogger(__name__)

import torch.nn as nn

class ConcatPoolingFusion(nn.Module):
    def __init__(self, d1, d2, d3):
        super(ConcatPoolingFusion, self).__init__()
        # Define a fully connected layer to model the interaction between the concatenated features
        self.fc = nn.Linear(d1 + d2, d3)

    def forward(self, x1, x2):
        # x1: (b, l1, d1)
        # x2: (b, l2, d2)
        b, l1, d1 = x1.size()
        _, l2, d2 = x2.size()

        # Repeat x1 along l2 dimension and x2 along l1 dimension
        x1 = x1.unsqueeze(2).repeat(1, 1, l2, 1)  # (b, l1, l2, d1)
        x2 = x2.unsqueeze(1).repeat(1, l1, 1, 1)  # (b, l1, l2, d2)

        # Concatenate along the feature dimension
        concat_features = torch.cat((x1, x2), dim=-1)  # (b, l1, l2, d1 + d2)

        # Apply the fully connected layer
        output = self.fc(concat_features)  # (b , l1 , l2, d3)

        # Reshape to the desired output
        output = output.mean(dim=2)  # (b, l1, d3)

        return output

class AddPoolingFusion(nn.Module):
    def __init__(self, d1, d2, d3):
        super(AddPoolingFusion, self).__init__()
        # Define a fully connected layer to model the interaction after addition
        self.fc1 = nn.Linear(d1, d3)
        self.fc2 = nn.Linear(d2, d3)

    def forward(self, x1, x2):
        # x1: (b, l1, d1)
        # x2: (b, l2, d2)
        b, l1, d1 = x1.size()
        _, l2, d2 = x2.size()

        # Repeat x1 along l2 dimension and x2 along l1 dimension
        x1 = x1.unsqueeze(2).repeat(1, 1, l2, 1)  # (b, l1, l2, d1)
        x2 = x2.unsqueeze(1).repeat(1, l1, 1, 1)  # (b, l1, l2, d2)

        # Map to d3
        x1 = self.fc1(x1)  # (b, l1, l2, d3)
        x2 = self.fc2(x2)  # (b, l1, l2, d3)

        # Add
        output = x1 + x2  # (b, l1, l2, d3)        

        # Reshape to the desired output
        output = output.mean(dim=2)  # (b, l1, d3)

        return output

class BilinearPoolingFusion(nn.Module):
    def __init__(self, d1, d2, d3):
        super(BilinearPoolingFusion, self).__init__()
        # Define a bilinear layer to model the interaction between the two modalities
        self.bilinear = nn.Bilinear(d1, d2, d3)

    def forward(self, x1, x2):
        # x1: (b, l1, d1)
        # x2: (b, l2, d2)
        b, l1, d1 = x1.size()
        _, l2, d2 = x2.size()

        # Repeat x1 along l2 dimension and x2 along l1 dimension
        x1 = x1.unsqueeze(2).repeat(1, 1, l2, 1)  # (b, l1, l2, d1)
        x2 = x2.unsqueeze(1).repeat(1, l1, 1, 1)  # (b, l1, l2, d2)

        # Reshape for bilinear pooling
        x1 = x1.contiguous().view(-1, d1)  # (b * l1 * l2, d1)
        x2 = x2.contiguous().view(-1, d2)  # (b * l1 * l2, d2)

        # Apply bilinear transformation
        output = self.bilinear(x1, x2)  # (b * l1 * l2, d3)

        # Reshape to the desired output
        output = output.view(b, l1, l2, -1).mean(dim=2)  # (b, l1, d3)

        return output

class TokenClassificationHead(nn.Module):
    """Head for token-level classification tasks."""

    def __init__(
        self,
        pocket_feat_dim,
        ligand_feat_dim,
        inner_dim,
        cross_attention_heads,
        cross_attention_layers,
        num_classes,
        activation_fn,
        dropout,
        weighted_center,
        merge_emb_method=2,
    ):
        super().__init__()
        
        def get_activation_layer(activation_fn):
            if activation_fn == "relu":
                return nn.ReLU()
            elif activation_fn == "gelu":
                return nn.GELU()
            elif activation_fn == "tanh":
                return nn.Tanh()
            else:
                raise RuntimeError("--activation-fn {} not supported".format(activation_fn))
        
        self.cross_attention_layers_num = cross_attention_layers
        self.merge_emb_method = merge_emb_method
        if merge_emb_method == 2:
            self.cross_attention_layers = nn.ModuleList(
                [
                    CrossAttention(query_dim=pocket_feat_dim, cross_attention_dim=ligand_feat_dim,
                                    heads=cross_attention_heads, dim_head=inner_dim//cross_attention_heads,
                                    dropout=dropout,bias=True,
                                    upcast_attention=False,upcast_softmax=False,
                                    added_kv_proj_dim=None,norm_num_groups=None,
                                    processor=None)
                    for _ in range(self.cross_attention_layers_num)
                ]
            )
        elif merge_emb_method == 1:
            self.merge_emb_layer = ConcatPoolingFusion(pocket_feat_dim, ligand_feat_dim, inner_dim)
        elif merge_emb_method == 0:
            self.merge_emb_layer = AddPoolingFusion(pocket_feat_dim, ligand_feat_dim, inner_dim)
        else:
            raise NotImplementedError
        
        self.reactive_predict_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(inner_dim, inner_dim),
            get_activation_layer(activation_fn),
            nn.Dropout(p=dropout),
            nn.Linear(inner_dim, 1)
        )
        self.weighted_center = weighted_center
        
        if self.weighted_center == 0:
            self.pocket_pred_head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(inner_dim, inner_dim),
                get_activation_layer(activation_fn),
                nn.Dropout(p=dropout),
                nn.Linear(inner_dim, 3)
            ) # directly predict 3d coords
        else:
            self.pocket_pred_head = copy.deepcopy(self.reactive_predict_head) # same structure as reactive_predict_head

    def forward(self, protein_feat, protein_coord, ligand_feat, attention_mask, pocket_padding_mask,**kwargs):
        protein_res_mask = ~pocket_padding_mask

        if self.merge_emb_method == 1: # default is 2
            complex_emb = self.merge_emb_layer(protein_feat, ligand_feat)
        elif self.merge_emb_method == 0:
            complex_emb = self.merge_emb_layer(protein_feat, ligand_feat)
        elif self.merge_emb_method == 2:
            complex_emb = protein_feat
            for i in range(self.cross_attention_layers_num):
                complex_emb = self.cross_attention_layers[i](complex_emb, ligand_feat,  attention_mask=attention_mask) # (b, n, d)
        
        # get reactive pred
        reactive_logit = self.reactive_predict_head(complex_emb + protein_feat).squeeze(-1) # (b,n)
        reactive_logit = reactive_logit * protein_res_mask
        
        # get pocket center
        # pocket_weight = pocket_weight * protein_res_mask.unsqueeze(-1)
        # pred_pocket_center = pocket_weight * protein_coord # FIXME: use softmax or sigmoid to active
        # pred_pocket_center = pred_pocket_center.sum(dim=1) / protein_res_mask.sum(dim=1)
        
        if self.weighted_center == 0: # default value is 2
            pocket_coords = self.pocket_pred_head(complex_emb[:,0,:]) # (b, 3) pred using cls
            pocket_coords = torch.sigmoid(pocket_coords)
            radius = torch.linalg.vector_norm(protein_coord,dim=-1).max(dim=1)[0] # max distance to original coord
            pred_pocket_center = pocket_coords * radius # (b, 3)
        else:
            pocket_weight = self.pocket_pred_head(complex_emb + protein_feat).squeeze(-1) # (b, n) 
            if self.weighted_center == 1:
                pocket_weight = torch.nn.functional.softmax(pocket_weight, dim=1) # (b,n)
                pred_pocket_center = pocket_weight.unsqueeze(-1) * protein_coord # (b,n,3)
                pred_pocket_center = pred_pocket_center.sum(dim=1) # (b,3)
            elif self.weighted_center == 2:
                pocket_weight = torch.sigmoid(pocket_weight) # (b,n)
                pred_pocket_center = pocket_weight.unsqueeze(-1) * protein_coord # (b,n,3)
                pred_pocket_center = pred_pocket_center.sum(dim=1) / pocket_weight.sum(dim=1) # (b,3)
            else:
                raise NotImplementedError
            
        
        return pred_pocket_center, reactive_logit

@register_model("covDocker_reactive_site_model")
class CovDockerReactiveSiteModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--ligand_feat_dim",
            type=int,
            default=512,
            help="ligand feature dimension",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--cross-attention-heads",
            type=int,
            default=8,
            help="number of cross attention heads",
        )
        parser.add_argument(
            "--cross-attention-layers",
            type=int,
            default=2,
            help='number of cross attention layers'
        )
        parser.add_argument(
            "--pocket-coord-huber-delta",
            type=float,
            default=3.0,
            help="delta for huber loss for pocket center coord prediction"
        )
        parser.add_argument(
            "--pocket-coord-loss-weight",
            type=float,
            default=0.3,
            help="weight for pocket center coord prediction loss"
        )
        parser.add_argument(
            "--pocket-token-clf-loss-weight",
            type=float,
            default=1.0,
            help="weight for pocket classification loss"
        )
        
        parser.add_argument(
            "--reactive-loss-weight",
            type=float,
            default=1.0,
            help="weight for reactive aa prediction loss"
        )
        parser.add_argument(
            "--weighted-center",
            type=int,
            default=2,
            help='0: directly predict center coords using cls; 1: softmax type weighted center; 2: sigmoid type weighted center'
        )
        
        parser.add_argument(
            "--merge-emb-method",
            type=int,
            default=2,
            help='0: add -> mean pooling; 1: concat -> mean pooling; 2: cross attention'
        )

    def __init__(self, args, mol_dictionary, protein_dictionary):
        super().__init__()
        CovDockerReactiveSiteModel_architecture(args)
        self.args = args
        self.padding_idx = protein_dictionary.pad()
        self.protein_model = UniMolModel(self.args, protein_dictionary)
        self.mol_model = UniMolModel(self.args.mol, mol_dictionary)
        # self.out_proj = TransformerEncoderLayer(
        #             embed_dim=args.encoder_embed_dim,
        #             ffn_embed_dim=args.encoder_ffn_embed_dim,
        #             attention_heads=args.encoder_attention_heads,
        #             dropout=args.dropout,
        #             attention_dropout=args.attention_dropout,
        #             activation_dropout=args.activation_dropout,
        #             activation_fn=args.activation_fn,
        #         )
        self.token_clf_head = TokenClassificationHead(
                            pocket_feat_dim=self.args.encoder_embed_dim,
                            ligand_feat_dim=self.args.ligand_feat_dim,
                            inner_dim=self.args.encoder_embed_dim,
                            cross_attention_heads=self.args.cross_attention_heads,
                            cross_attention_layers=self.args.cross_attention_layers,
                            num_classes=1, # is bonded aa or not
                            activation_fn=self.args.pooler_activation_fn,
                            dropout=self.args.pooler_dropout,
                            weighted_center = args.weighted_center,
                            merge_emb_method=args.merge_emb_method
                            )
        
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.mol_dictionary,  task.protein_dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        mol_src_tokens,
        mol_src_distance,
        mol_src_edge_type,
        masked_tokens=None,
        features_only=True,
        **kwargs
    ):
        def get_dist_features(dist, et, flag):
            if flag == "mol":
                n_node = dist.size(-1)
                gbf_feature = self.mol_model.gbf(dist, et)
                gbf_result = self.mol_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias
            elif flag == "protein":
                n_node = dist.size(-1)
                gbf_feature = self.protein_model.gbf(dist, et)
                gbf_result = self.protein_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
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
        protein_padding_mask = src_tokens.eq(self.protein_model.padding_idx)
        protein_x = self.protein_model.embed_tokens(src_tokens)
        protein_graph_attn_bias = get_dist_features(
            src_distance, src_edge_type, "protein"
        )
        protein_outputs = self.protein_model.encoder(
            protein_x, padding_mask=protein_padding_mask, attn_mask=protein_graph_attn_bias
        )
        protein_encoder_rep = protein_outputs[0]
        protein_encoder_pair_rep = protein_outputs[1]


        mol_sz = mol_encoder_rep.size(1)
        pocket_sz = protein_encoder_rep.size(1)
        
        def fill_cross_attn_mask(protein_padding_mask, mol_padding_mask):
            complex_padding_mask = torch.einsum('bl,bt->blt', protein_padding_mask, mol_padding_mask).to(protein_padding_mask.dtype)
            return complex_padding_mask.masked_fill_(complex_padding_mask == 1,float("-inf"))

        pred_pocket_center, reactive_logit = self.token_clf_head(protein_encoder_rep, src_coord, mol_encoder_rep ,fill_cross_attn_mask(protein_padding_mask, mol_padding_mask), protein_padding_mask)
        # pred = torch.argmax(x.squeeze(),dim=-1) # (b,n)
        # return pred
        return pred_pocket_center, reactive_logit


@register_model_architecture("covDocker_reactive_site_model", "covDocker_reactive_site_model_large")
def CovDockerReactiveSiteModel_architecture(args):
    def base_architecture(args):
        parser = argparse.ArgumentParser()
        args.mol = parser.parse_args([])
    
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
    
        args.encoder_layers = getattr(args, "encoder_layers", 25)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
        args.dropout = getattr(args, "dropout", 0.1)
        args.emb_dropout = getattr(args, "emb_dropout", 0.1)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.activation_dropout = getattr(args, "activation_dropout", 0.0)
        args.pooler_dropout = getattr(args, "pooler_dropout", 0.1)
        args.max_seq_len = getattr(args, "max_seq_len", 1024)
        args.activation_fn = getattr(args, "activation_fn", "gelu")
        args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
        args.post_ln = getattr(args, "post_ln", False)
        args.masked_coord_loss = getattr(args, "masked_coord_loss", 0.0)
        args.masked_dist_loss = getattr(args, "masked_dist_loss", 0.0)
        args.masked_token_loss = getattr(args,"masked_token_loss", 0.0)

    base_architecture(args)



@register_model_architecture("covDocker_reactive_site_model", "covDocker_reactive_site_model_base")
def CovDockerReactiveSiteModel_architecture(args):
    def base_architecture(args):
        parser = argparse.ArgumentParser()
        args.mol = parser.parse_args([])
    
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
    
        args.encoder_layers = getattr(args, "encoder_layers", 15)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
        args.dropout = getattr(args, "dropout", 0.1)
        args.emb_dropout = getattr(args, "emb_dropout", 0.1)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.activation_dropout = getattr(args, "activation_dropout", 0.0)
        args.pooler_dropout = getattr(args, "pooler_dropout", 0.1)
        args.max_seq_len = getattr(args, "max_seq_len", 1024)
        args.activation_fn = getattr(args, "activation_fn", "gelu")
        args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
        args.post_ln = getattr(args, "post_ln", False)
        args.masked_coord_loss = getattr(args, "masked_coord_loss", 0.0)
        args.masked_dist_loss = getattr(args, "masked_dist_loss", 0.0)
        args.masked_token_loss = getattr(args,"masked_token_loss", 0.0)

    base_architecture(args)


@register_model_architecture("covDocker_reactive_site_model", "covDocker_reactive_site_model_sm")
def CovDockerReactiveSiteModel_architecture(args):

    def base_architecture(args):
        parser = argparse.ArgumentParser()
        args.mol = parser.parse_args([])
    
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
        
        args.encoder_layers = getattr(args, "encoder_layers", 5)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
        args.dropout = getattr(args, "dropout", 0.1)
        args.emb_dropout = getattr(args, "emb_dropout", 0.1)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.activation_dropout = getattr(args, "activation_dropout", 0.0)
        args.pooler_dropout = getattr(args, "pooler_dropout", 0.1)
        args.max_seq_len = getattr(args, "max_seq_len", 1024)
        args.activation_fn = getattr(args, "activation_fn", "gelu")
        args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
        args.post_ln = getattr(args, "post_ln", False)
        args.masked_coord_loss = getattr(args, "masked_coord_loss", 0.0)
        args.masked_dist_loss = getattr(args, "masked_dist_loss", 0.0)
        args.masked_token_loss = getattr(args,"masked_token_loss", 0.0)

    base_architecture(args)
