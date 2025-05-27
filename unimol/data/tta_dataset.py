# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class TTADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, conf_size=10):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[smi_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])
        smi = self.dataset[smi_idx]["smi"]
        target = self.dataset[smi_idx]["target"]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "smi": smi,
            "target": target,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class TTADockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        holo_coordinates,
        holo_pocket_coordinates,
        is_train=True,
        conf_size=10,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.is_train = is_train
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        idx2atom = {
            5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 14:'Si', 15:'P', 16:'S', 17:'Cl', 34:'Se', 35:'Br', 53:'I' 
        }
        atoms = np.array([idx2atom[a] for a in self.dataset[index][self.atoms]])
        coordinates = np.array(self.dataset[index][self.coordinates])
        pocket_atoms = np.array([idx2atom[a] for a in self.dataset[index][self.pocket_atoms]])
        pocket_coordinates = np.array(self.dataset[index][self.pocket_coordinates])
        if self.is_train:
            holo_coordinates = np.array(self.dataset[index][self.holo_coordinates])
            holo_pocket_coordinates = np.array(self.dataset[index][self.holo_pocket_coordinates])
        else:
            holo_coordinates = coordinates
            holo_pocket_coordinates = pocket_coordinates

        smi = self.dataset[smi_idx]["smi"]
        pocket = self.dataset[smi_idx]["pocket"]

        return {
            "atoms": atoms.astype(np.str_),
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms.astype(np.str_),
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
