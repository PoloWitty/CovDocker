import numpy as np
import torch
from functools import lru_cache

from unicore.data import BaseWrapperDataset
from scipy.sparse import coo_matrix
import torch.distributed as dist
from joblib import Memory
from numba import njit

UNREACHABLE_NODE_DISTANCE = 510


class MaskDataset(BaseWrapperDataset):
    def __init__(self, dataset, mask_value):
        super().__init__(dataset)
        self.dataset = dataset
        self.mask_value = mask_value

    def set_epoch(self, epoch, **unused):
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)
    
    @lru_cache(maxsize=16)
    def __cached_item__(self, idx: int, epoch: int):
        item = self.dataset[idx]
        item = torch.full_like(item, self.mask_value)
        return item

    def __getitem__(self, idx):
        return self.__cached_item__(idx, self.epoch)

class ConcatDataset(BaseWrapperDataset):

    def __init__(self, ligand_dataset, pocket_dataset):
        self.ligand_dataset = ligand_dataset
        self.pocket_dataset = pocket_dataset

        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        self.ligand_dataset.set_epoch(epoch)
        self.pocket_dataset.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.ligand_dataset)
    
    @lru_cache(maxsize=16)
    def __cached_item__(self, idx: int, epoch: int):
        ligand_item = self.ligand_dataset[idx]
        pocket_item = self.pocket_dataset[idx]

        item = torch.cat([ligand_item, pocket_item], dim=0)
        return item

    def __getitem__(self, idx):
        return self.__cached_item__(idx, self.epoch)
    
class Concat2dDataset(BaseWrapperDataset):
    def __init__(self, ligand2d_dataset, pocket2d_dataset, pad_token ):
        self.ligand2d_dataset = ligand2d_dataset
        self.pocket2d_dataset = pocket2d_dataset
        self.pad_token = pad_token
        
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        self.ligand2d_dataset.set_epoch(epoch)
        self.pocket2d_dataset.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.ligand2d_dataset)
    
    @lru_cache(maxsize=16)
    def __cached_item__(self, idx: int, epoch: int):
        ligand2d_item = self.ligand2d_dataset[idx]
        pocket2d_item = self.pocket2d_dataset[idx]
        
        ligand2d_sz = ligand2d_item.size(0)
        pocket2d_sz = pocket2d_item.size(0)
        assert ligand2d_item.size(0) == ligand2d_item.size(1)
        assert pocket2d_item.size(0) == pocket2d_item.size(1)
        concat2d_item = torch.full((ligand2d_sz + pocket2d_sz, ligand2d_sz + pocket2d_sz), self.pad_token).type_as(ligand2d_item)
        
        # fill values
        concat2d_item[:ligand2d_sz, :ligand2d_sz] = ligand2d_item.contiguous()
        concat2d_item[-pocket2d_sz:, -pocket2d_sz:] = pocket2d_item.contiguous()
        return concat2d_item
    
    def __getitem__(self, idx):
        return self.__cached_item__(idx, self.epoch)
    
class Concat2dCrossDataset(BaseWrapperDataset):
    def __init__(self, ligand2d_dataset, cross2d_dataset, pad_token ):
        self.ligand2d_dataset = ligand2d_dataset
        self.cross2d_dataset = cross2d_dataset
        self.pad_token = pad_token
        
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        self.ligand2d_dataset.set_epoch(epoch)
        self.cross2d_dataset.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.ligand2d_dataset)
    
    @lru_cache(maxsize=16)
    def __cached_item__(self, idx: int, epoch: int):
        ligand2d_item = self.ligand2d_dataset[idx]
        cross2d_item = self.cross2d_dataset[idx]
        
        ligand2d_sz = ligand2d_item.size(1)
        cross2d_sz = cross2d_item.size(1)
        assert ligand2d_item.size(0) == ligand2d_item.size(1)
        assert cross2d_item.size(0) == ligand2d_item.size(0)
        concat2d_item = torch.full((ligand2d_sz, ligand2d_sz + cross2d_sz), self.pad_token).type_as(ligand2d_item)
        
        # fill values
        concat2d_item[:, :ligand2d_sz] = ligand2d_item.contiguous()
        concat2d_item[:, -cross2d_sz:] = cross2d_item.contiguous()
        return concat2d_item
    
    def __getitem__(self, idx):
        return self.__cached_item__(idx, self.epoch)

class BondDataset(BaseWrapperDataset):
    '''adj bond dataset'''
    def __init__(self, ligand_bonds_dataset, pocket_bonds_dataset, inter_bonds_dataset, use_special_inter_bond_idx=False):
        self.ligand_bonds_dataset = ligand_bonds_dataset
        self.pocket_bonds_dataset = pocket_bonds_dataset
        self.inter_bonds_dataset = inter_bonds_dataset
        self.use_special_inter_bond_idx = use_special_inter_bond_idx
        
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        self.ligand_bonds_dataset.set_epoch(epoch)
        self.pocket_bonds_dataset.set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.ligand_bonds_dataset)
    
    @lru_cache(maxsize=16)
    def __cached_item__(self, idx: int, epoch: int):
        ligand_bonds = self.ligand_bonds_dataset[idx]
        pocket_bonds = self.pocket_bonds_dataset[idx]
        inter_bond = self.inter_bonds_dataset[idx]
        
        # build adjacency matrix
        ligand_sz = ligand_bonds.max() + 1
        ligand_adj = coo_matrix((ligand_bonds[:, 2], (ligand_bonds[:, 0], ligand_bonds[:, 1])),shape=(ligand_sz, ligand_sz)).toarray()
        pocket_sz = pocket_bonds.max() + 1
        pocket_adj = coo_matrix((pocket_bonds[:, 2], (pocket_bonds[:, 0], pocket_bonds[:, 1])),shape=(pocket_sz, pocket_sz)).toarray()
        docking_adj = np.zeros((1 + ligand_sz + 2 + pocket_sz + 1, 1 + ligand_sz + 2 + pocket_sz + 1), dtype=np.int64) # bos + ligand + eos + bos + pocket + eos
        docking_adj[1:ligand_sz+1, 1:ligand_sz+1] = ligand_adj
        docking_adj[-pocket_sz-1:-1, -pocket_sz-1:-1] = pocket_adj
        # add inter bond [bond_pocket_idx, bond_ligand_idx]
        if self.use_special_inter_bond_idx:
            docking_adj[inter_bond[1]+1, ligand_sz + 3 +inter_bond[0]] = 4
            docking_adj[ligand_sz + 3 +inter_bond[0], inter_bond[1]+1] = 4
        else:
            docking_adj[inter_bond[1]+1, ligand_sz + 3 +inter_bond[0]] = 1
            docking_adj[ligand_sz + 3 +inter_bond[0], inter_bond[1]+1] = 1
        
        edge_input = torch.from_numpy(docking_adj)
        return edge_input
    
    def __getitem__(self, idx):
        return self.__cached_item__(idx, self.epoch)

@njit
def floyd_warshall(M):
    (nrows, ncols) = M.shape
    assert nrows == ncols
    n = nrows
    # set unreachable nodes distance to UNREACHABLE_NODE_DISTANCE
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                M[i, j] = UNREACHABLE_NODE_DISTANCE

    for i in range(n):
        M[i, i] = 0

    # floyed algo
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj

    for i in range(n):
        for j in range(n):
            if M[i, j] >= UNREACHABLE_NODE_DISTANCE:
                M[i, j] = UNREACHABLE_NODE_DISTANCE
    return M

class SPDDataset(BaseWrapperDataset):
    '''shortest path distance dataset'''
    def __init__(self, bond_dataset):
        self.dataset = bond_dataset # bond dataset is dataset of adjacency matrix

        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)
    
    @lru_cache(maxsize=16)
    def __cached_item__(self, idx: int, epoch: int):
        bond_adj = self.dataset[idx]
        
        shortest_path_distance = floyd_warshall(bond_adj.numpy().astype(np.bool_).astype(np.int)) # use unweighted adj matrix to compute shortest path
        
        shortest_path_distance = torch.from_numpy(shortest_path_distance + 1) # leave 0 for pad
        return shortest_path_distance

    def __getitem__(self, idx):
        return self.__cached_item__(idx, self.epoch)

class RightPadDataset2DFeature(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collate_2d_feat(
        self,
        values,
        pad_idx,
        left_pad=False,
        pad_to_length=None,
        pad_to_multiple=1,
        ):
        """Convert a list of (n,n, ...) tensors into a padded (n+pad,n+pad, ...) tensor."""
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)
        if pad_to_multiple != 1 and size % pad_to_multiple != 0:
            size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
        bs = len(values)
        feat_size = values[0][0,0].shape
        res = values[0].new(bs, size, size, *feat_size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):, size - len(v):] if left_pad else res[i][:len(v), :len(v)])
        return res

    def collater(self, samples):
        return self.collate_2d_feat(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
