"""
desc:	dataset class for covDocker reactive site prediction
author:	Yangzhe Peng
date:	2024/05/12
"""


import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset

class CovDockerReactiveSitePredictionDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        ligand_atoms,
        ligand_coords,
        protein_residue,
        protein_CA_coords,
        pocket_mask,
        pdbid,
        smi,
        target,
    ):
        self.dataset = dataset
        self.ligand_atoms = ligand_atoms
        self.ligand_coords = ligand_coords
        self.protein_residue = protein_residue
        self.protein_CA_coords = protein_CA_coords
        self.pocket_mask = pocket_mask
        self.pdbid = pdbid
        self.smi = smi
        self.target = target
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        ligand_atoms = np.array(self.dataset[index][self.ligand_atoms])
        size = len(self.dataset[index][self.ligand_coords])
        ligand_coords = np.array(self.dataset[index][self.ligand_coords])

        protein_residue = np.array(self.dataset[index][self.protein_residue])
        protein_CA_coords = np.array(self.dataset[index][self.protein_CA_coords])
        
        pocket_mask = np.array(self.dataset[index][self.pocket_mask])
        

        smi = self.dataset[index][self.smi]
        pdbid = self.dataset[index][self.pdbid]
        target = self.dataset[index][self.target]

        return {
            "ligand_atoms": ligand_atoms.astype(np.str_),
            "ligand_coords": ligand_coords.astype(np.float32),
            "protein_residue": protein_residue.astype(np.str_),
            "protein_CA_coords": protein_CA_coords.astype(np.float32),
            "pocket_mask": pocket_mask.astype(np.bool_),
            "smi": smi,
            "pdbid": pdbid,
            "target": target
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
