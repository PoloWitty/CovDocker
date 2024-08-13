# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class RemoveHydrogenDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        remove_hydrogen=False,
        remove_polar_hydrogen=False,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class RemoveHydrogenResiduePocketDataset(BaseWrapperDataset):
    def __init__(self, dataset, atoms, residues, coordinates, remove_hydrogen=True):
        self.dataset = dataset
        self.atoms = atoms
        self.residues = residues
        self.coordinates = coordinates
        self.remove_hydrogen = remove_hydrogen
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        residues = dd[self.residues]
        coordinates = dd[self.coordinates]
        if len(atoms) != len(residues):
            min_len = min(len(atoms), len(residues))
            atoms = atoms[:min_len]
            residues = residues[:min_len]
            coordinates = coordinates[:min_len, :]

        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            atoms = atoms[mask_hydrogen]
            residues = residues[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]

        dd[self.atoms] = atoms
        dd[self.residues] = residues
        dd[self.coordinates] = coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class RemoveHydrogenPocketDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        holo_coordinates,
        bonds = None,
        remove_hydrogen=True,
        remove_polar_hydrogen=False,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.holo_coordinates = holo_coordinates
        self.remove_hydrogen = remove_hydrogen
        self.remove_polar_hydrogen = remove_polar_hydrogen
        self.bonds = bonds
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        atoms = dd[self.atoms]
        coordinates = dd[self.coordinates]
        holo_coordinates = dd[self.holo_coordinates]
        if self.bonds != None:
            bonds = dd[self.bonds]
        
        H_symbol = 'H' if type(atoms[0]) == str else 1 # 1 stand for hydrogen's atomic number

        if self.remove_hydrogen:
            mask_hydrogen = atoms != H_symbol
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
            holo_coordinates = holo_coordinates[mask_hydrogen]
            if self.bonds != None:
                assert not self.remove_hydrogen, "remove_hydrogen must be False when bonds is not None"
        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != H_symbol:
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]
                holo_coordinates = holo_coordinates[:-end_idx]
                if self.bonds != None:
                    bonds = bonds[mask_hydrogen]
        dd[self.atoms] = atoms
        dd[self.coordinates] = coordinates.astype(np.float32)
        dd[self.holo_coordinates] = holo_coordinates.astype(np.float32)
        if self.bonds != None:
            dd[self.bonds] = bonds.astype(np.int16)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
