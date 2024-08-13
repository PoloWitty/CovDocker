"""
desc:	covDocker step1 task: bonded aa position prediction
author:	Yangzhe Peng
date:	2024/01/15
"""

import logging
import os
import pdb

from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    RawLabelDataset,
    FromNumpyDataset,
    EpochShuffleDataset,
)

from unimol.data import (
    KeyDataset,
    ConformerSamplePocketFinetuneDataset,
    CovDockerConformerSamplePocketDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDatasetCoord,
    RightPadDatasetCross2D,
    AddValueDataset,
    CroppingResiduePocketDataset,
    RemoveHydrogenResiduePocketDataset,
    FromStrLabelDataset,
    CovDockerReactiveSitePredictionDataset
)

from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils


logger = logging.getLogger(__name__)

three_to_one = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','MSE','PHE','PRO','PYL','SER','SEC','THR','TRP','TYR','VAL','ASX','GLX','XAA','XLE']

# same feature as diffdock's pdbbind_lm_embedding_preparation.py feature
allowable_features = {'possible_amino_acids': three_to_one + ['misc']}

@register_task("covDocker_reactive_site")
class CovDockerReactiveSitePredictionTask(UnicoreTask):
    """Task for training cov docking position models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="cov docking position prediction data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=2,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict_pkt.txt",
            help="dictionary file",
        )

    def __init__(self, args, mol_dictionary, protein_dictionary):
        super().__init__(args)
        self.protein_dictionary = protein_dictionary
        self.mol_dictionary = mol_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = protein_dictionary.add_symbol("[MASK]", is_special=True)
        self.ligand_mask_idx = mol_dictionary.add_symbol("[MASK]", is_special=True)
        self.mean = None
        self.std = None


    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        dict_pkt_filename = os.path.join(args.data,'dict_pkt.txt')
        if not os.path.exists(dict_pkt_filename):
            dict_list = ['[PAD]','[CLS]','[SEP]','[UNK]']
            dict_list += allowable_features['possible_amino_acids'][:-1] # skip misc
            with open(dict_pkt_filename,'wb') as fp:
                for d in dict_list:
                    fp.write((d+'\n').encode('utf-8'))
        
        protein_dictionary = Dictionary.load(dict_pkt_filename)
        logger.info("protein dictionary: {} types".format(len(protein_dictionary)))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        return cls(args, mol_dictionary, protein_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        pdb_dataset = KeyDataset(dataset, "pdbid")
        res_ids_dataseet = KeyDataset(dataset, "res_ids")
        dataset = CovDockerReactiveSitePredictionDataset(
            dataset,
            "pre_reactive_ligand_atoms",
            "pre_reactive_ligand_coords",
            "residue",
            "protein_CA_coords",
            "pocket_mask",
            "pdbid",
            "smiles",
            "target"
        )
        dataset = NormalizeDataset(dataset, "protein_CA_coords")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        # target
        reactive_tgt_dataset = KeyDataset(dataset, "target")
        reactive_tgt_dataset = RawLabelDataset(reactive_tgt_dataset)
        reactive_tgt_dataset = AddValueDataset(reactive_tgt_dataset, 1) # 1 for bos token
        pocket_tgt_dataset = KeyDataset(dataset, "pocket_mask")
        pocket_tgt_dataset = FromNumpyDataset(pocket_tgt_dataset)
        pocket_tgt_dataset = PrependAndAppend(
            pocket_tgt_dataset, 0, 0
        )
        
        # protein part
        src_dataset = KeyDataset(dataset, "protein_residue")
        src_dataset = TokenizeDataset(
            src_dataset, self.protein_dictionary, max_seq_len=self.args.max_seq_len
        )
        src_dataset = PrependAndAppend(
            src_dataset, self.protein_dictionary.bos(), self.protein_dictionary.eos()
        )
        coord_dataset = KeyDataset(dataset, "protein_CA_coords")
        edge_type = EdgeTypeDataset(src_dataset, len(self.protein_dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0) # position for bos and eos
        distance_dataset = DistanceDataset(coord_dataset)
        
        # ligand part
        ligand_src_dataset = KeyDataset(dataset, "ligand_atoms")
        ligand_src_dataset = TokenizeDataset(
            ligand_src_dataset, self.mol_dictionary, max_seq_len=self.args.max_seq_len
        )
        ligand_src_dataset = PrependAndAppend(
            ligand_src_dataset, self.mol_dictionary.bos(), self.mol_dictionary.eos()
        )
        ligand_coord_dataset = KeyDataset(dataset, "ligand_coords")
        ligand_edge_type = EdgeTypeDataset(ligand_src_dataset, len(self.mol_dictionary))
        ligand_coord_dataset = FromNumpyDataset(ligand_coord_dataset)
        ligand_coord_dataset = PrependAndAppend(ligand_coord_dataset, 0.0, 0.0) # position for bos and eos
        ligand_distance_dataset = DistanceDataset(ligand_coord_dataset)
        
        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.protein_dictionary.pad(),
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                    "mol_src_tokens": RightPadDataset(
                        ligand_src_dataset,
                        pad_idx=self.mol_dictionary.pad(),
                    ),
                    "mol_src_distance": RightPadDataset2D(
                        ligand_distance_dataset,
                        pad_idx=0,
                    ),
                    "mol_src_edge_type": RightPadDataset2D(
                        ligand_edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": reactive_tgt_dataset,
                    "pocket_target": RightPadDataset(
                        pocket_tgt_dataset,
                        pad_idx=0
                    ),
                },
                "pdbid": RawArrayDataset(pdb_dataset),
                "res_ids": RawArrayDataset(res_ids_dataseet),
            },
        )

        if split.startswith("train"):
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.args.seed
            )
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        """Build a new model instance."""
        from unicore import models
    
        model = models.build_model(args, self)
        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
            
        return model