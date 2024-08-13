
import pdb
from typing import List, Tuple
from collections import Counter
import pickle

import numpy as np

import networkx as nx
from rdkit import Chem
from scipy import optimize

# copy from https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def nx_to_mol(G):
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol

# end copy

def get_match_count(list1, list2):
    counter1 = Counter(list1); counter2 = Counter(list2)
    return sum((counter1 & counter2).values())
    # return sum([1 for i in set(list2) if i in list1])

def build_edge_list(G:nx.Graph, src_idx:int, neighbor_idxes: List[int]):
    edges = set(G.edges(sorted(neighbor_idxes)))
    rtn_edges = []
    for (s,e) in edges:
        rtn_edges.append(tuple(
            sorted(
                (G.nodes[s]["atomic_num"],G.nodes[e]["atomic_num"]) 
            ) + 
            sorted(
                (G.nodes[s]["hybridization"],G.nodes[e]["hybridization"])
            ) +
            [int(G.edges[s,e]["bond_type"])] + 
            [max(nx.dijkstra_path_length(G, src_idx, s),nx.dijkstra_path_length(G, src_idx, e))]
        ))
    return rtn_edges

def cost_mtx(A: nx.Graph, B: nx.Graph, K: int=1) -> np.ndarray:
    assert K>=1, "K must be greater than or equal to 1"
    rtn_mat = np.zeros((len(A.nodes()), len(B.nodes())))
    cost_fn = lambda match, len_i, len_j, i_equ_j : match/(len_i+len_j-match) + i_equ_j

    for i in A.nodes():
        for j in B.nodes():
            # neighbor_i = [A.nodes[atom]["atomic_num"] for atom in nx.ego_graph(A, n=i, radius=K, center=False, undirected=True).nodes()]
            # neighbor_j = [B.nodes[atom]["atomic_num"] for atom in nx.ego_graph(B, n=j, radius=K, center=False, undirected=True).nodes()]
            i_equ_j = int(A.nodes[i]["atomic_num"]==B.nodes[j]["atomic_num"]  and A.nodes[i]["hybridization"]==B.nodes[j]["hybridization"])
            if not i_equ_j: # different type of atom, no need to compute to accelerate
                rtn_mat[i, j] = -1e-3
                continue
            neighbor_i = build_edge_list(A, i, nx.ego_graph(A, n=i, radius=K, center=False, undirected=True).nodes())
            neighbor_j = build_edge_list(B, j, nx.ego_graph(B, n=j, radius=K, center=False, undirected=True).nodes())
            if neighbor_i == [] and neighbor_j == []:
                rtn_mat[i, j] = -1e-3
            match_count = get_match_count(neighbor_i, neighbor_j)
            rtn_mat[i, j] = cost_fn(match_count, len(neighbor_i), len(neighbor_j), i_equ_j)
    return rtn_mat
    

# modify from https://spyrmsd.readthedocs.io/en/develop/_modules/spyrmsd/hungarian.html#cost_mtx
def optimal_assignment(A: nx.Graph, B: nx.Graph):
    """
    Solve the optimal assignment problems between atomic coordinates of
    molecules A and B.

    Parameters
    ----------
    A: nx.Graph
        Atomic coordinates of molecule A
    B: nx.Graph
        Atomic coordinates of molecule B

    Returns
    -------
    Tuple[float, nd.array, nd.array]
        Cost of the optimal assignment, together with the row and column
        indices of said assignment
    """

    C = cost_mtx(A, B, K=2)

    row_idx, col_idx = optimize.linear_sum_assignment(C, maximize=True)

    # Compute assignment cost
    cost = C[row_idx, col_idx].sum()

    return cost, row_idx, col_idx
# end modify

def draw_2dmol(mol, filename='tmp.png'):
    from rdkit.Chem import Draw
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    Draw.MolToFile(mol,filename,size=(600,600))

def remove_AA_backbone_from_product_smiles(product_smiles: str, aa_smiles: str, AA_backbone_smiles: str = 'NCC(=O)O') -> str:
    if product_smiles == "":
        return ""
    # the default parameter of AA_backbone_smiles is the smiles of GLY whose R group is H
    product_mol = Chem.MolFromSmiles(product_smiles); AA_mol = Chem.MolFromSmiles(aa_smiles)
    if product_mol is None or AA_mol is None:
        return ""
    
    AA_backbone_idx = list(AA_mol.GetSubstructMatch(Chem.MolFromSmarts(AA_backbone_smiles)))
    
    AA_idx = list(product_mol.GetSubstructMatch(AA_mol))
    if AA_idx == []:
        return ""
    
    edit_mol = Chem.EditableMol(product_mol)
    atoms_to_remove = [AA_idx[idx] for idx in AA_backbone_idx]
    edit_mol.BeginBatchEdit()
    for atom in atoms_to_remove:
        edit_mol.RemoveAtom(atom)
    edit_mol.CommitBatchEdit()
    product_mol_woBackbone = edit_mol.GetMol()
    Chem.SanitizeMol(product_mol_woBackbone)
    return Chem.MolToSmiles(product_mol_woBackbone)

def compute_substructure_accuracy(
    sampled_smiles: List[List[str]], target_smiles: List[str], top_Ks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computing top-K accuracy for each K in 'top_Ks'.
    """

    n_beams = np.max(
        np.array(
            [1, np.max(np.asarray([len(smiles) for smiles in sampled_smiles]))]
        )
    )
    top_Ks = top_Ks[top_Ks <= n_beams]
    n_Ks = len(top_Ks)

    accuracy = np.zeros(n_Ks)

    is_in_set = np.zeros((len(sampled_smiles), n_Ks), dtype=bool)
    for i_k, K in enumerate(top_Ks):
        for i_sample, mols in enumerate(sampled_smiles):
            pre_i_k = i_k -1 if i_k > 0 else 0
            pre_K = top_Ks[pre_i_k] if pre_i_k > 0 else 0
            for mol in mols[pre_K:K]:
                try:
                    if is_in_set[i_sample, pre_i_k] or is_in_set[i_sample, i_k]:
                        is_in_set[i_sample, i_k] = True
                        continue
                    mol = Chem.MolFromSmiles(mol)
                    target_mol = Chem.MolFromSmiles(target_smiles[i_sample])
                    if mol == None:
                        continue
                    is_in_set[i_sample, i_k] = mol.HasSubstructMatch(target_mol) and target_mol.HasSubstructMatch(mol)
                except:
                    pdb.set_trace()
    is_in_set = np.cumsum(is_in_set, axis=1)
    accuracy = np.mean(is_in_set > 0, axis=0)
    return accuracy, top_Ks

def compute_exact_match_accuracy(
    sampled_smiles: List[List[str]], target_smiles: List[str], top_Ks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computing top-K accuracy for each K in 'top_Ks'.
    """

    n_beams = np.max(
        np.array(
            [1, np.max(np.asarray([len(smiles) for smiles in sampled_smiles]))]
        )
    )
    top_Ks = top_Ks[top_Ks <= n_beams]
    n_Ks = len(top_Ks)

    accuracy = np.zeros(n_Ks)

    is_in_set = np.zeros((len(sampled_smiles), n_Ks), dtype=bool)
    for i_k, K in enumerate(top_Ks):
        for i_sample, mols in enumerate(sampled_smiles):
            top_K_mols = mols[0:K]

            if len(top_K_mols) == 0:
                continue
            is_in_set[i_sample, i_k] = target_smiles[i_sample] in top_K_mols

    is_in_set = np.cumsum(is_in_set, axis=1)
    accuracy = np.mean(is_in_set > 0, axis=0)
    return accuracy, top_Ks

def compute_ligand_acc(
    sampled_smiles: List[List[str]], target_smiles_woAA: List[str], top_Ks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    target_graphs = [mol_to_nx(Chem.MolFromSmiles(smiles.replace('Ti','H'))) for smiles in target_smiles_woAA] # Ti is aa sym
    
    sampled_molecules = [[Chem.MolFromSmiles(smi) for smi in smiles_list] for smiles_list in sampled_smiles ]
    sampled_graphs = [[mol_to_nx(mol) if mol!=None else None for mol in mols] for mols in sampled_molecules]

    n_beams = np.max(
        np.array(
            [1, np.max(np.asarray([len(smiles) for smiles in sampled_smiles]))]
        )
    ) # calc the actual sampler's beam size
    top_Ks = top_Ks[top_Ks <= n_beams]
    n_Ks = len(top_Ks)
    accuracy = np.zeros(n_Ks)
    is_in_set = np.zeros((len(sampled_smiles), n_Ks), dtype=bool)

    for i_sample, (sampled_graphs_topK,target_graph) in enumerate(zip(sampled_graphs,target_graphs)):
        for i_k, K in enumerate(top_Ks):
        # for sampled_graph in sampled_graphs_topK:
            sampled_graph = sampled_graphs_topK[K-1]
            if sampled_graph is None:
                continue
            _, row_idx, col_idx = optimal_assignment(target_graph, sampled_graph)
            entry_res = len(row_idx) == len(target_graph)
            is_in_set[i_sample, i_k] = entry_res
    is_in_set = np.cumsum(is_in_set, axis=1)
    accuracy = np.mean(is_in_set > 0, axis=0)
    return accuracy, top_Ks

if __name__=='__main__':
    targets = ['N[C@@H](CSCC(=O)N1CCC[C@@H](c2nc3ccccc3s2)C1)C(=O)O', 'CCN1CCN(Cc2ccc(NC(=O)c3ccc(C)c(Oc4ccnc(N[C@@H]5CCN(C(=O)CCSC[C@H](N)C(=O)O)C5)n4)c3)cc2C(F)(F)F)CC1', 'CCOC(=O)C[C@H](SC[C@H](N)C(=O)O)[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)OCc1ccccc1)[C@@H](C)OC(C)(C)C', 'N[C@@H](CSCC(=O)N1CCN(Cc2ccc(Cl)s2)CC1)C(=O)O', 'N[C@@H](CSCC(=O)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1)C(=O)O']
    smiles = [['N[C@@H](CSCC(=O)N1CCC[C@@H](c2nc3ccccc3s2)C1)C(=O)O'], ['CCN1CCN(Cc2ccc(NC(=O)c3ccc(C)c(Oc4ccnc(N[C@@H]5CCN(C(=O)CCSC[C@H](N)C(=O)O)C5)n4)c3)cc2C(F)(F)F)CC1'], ['CCOC(=O)C[C@@H](SC[C@H](N)C(=O)O)[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)OCc1ccccc1)[C@@H](C)OC(C)(C)C'], ['N[C@@H](CSCC(=O)N1CCN(Cc2ccc(Cl)s2)CC1)C(=O)O'], ['N[C@@H](CSCC(=O)N1CCN(S(=O)(=O)c2c(F)cccc2F)CC1)C(=O)O']]

    # only match the ligand part of the product smiles between target and predict
    # pred_mol = Chem.MolFromSmiles(smiles[0][0]) ; tar_mol = Chem.MolFromSmiles(targets[0])
    # pred_graph = mol_to_nx(pred_mol) ; tar_graph = mol_to_nx(tar_mol)
    # _, row_idx, col_idx = optimal_assignment(pred_graph, tar_graph)

    # entry_res = len(row_idx) / len(tar_graph)
    # print(entry_res)
    # print(row_idx)
    # print(col_idx)
    # print((smiles[0][0]))
    # print((targets[0]))
    
    # remove the AA backbone from product smiles
    # product_smiles = targets[0]
    # print(remove_AA_backbone_from_product_smiles(product_smiles))
    canonicalize_smiles = lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    # print(remove_AA_backbone_from_product_smiles('N[C@@H](CCC(=O)N[C@@H](CSCCS(=O)(=O)Oc1ccc(C[C@H](N)C(=O)O)cc1)C(=O)NCC(=O)O)C(=O)O', aa_smiles='N[C@@H](Cc1ccc(O)cc1)C(=O)O'))
    import pickle
    obj = pickle.load(open('sampled_test.pkl','rb'))
    sampled_smiles = obj['sampled_products']; target_smiles = obj['target_products']; reactants = obj['reactants']
    # target_smiles_woAAbackbone = obj['target_smiles_woAAbackbone'][10:30]; sampled_smiles_woAAbackbone = obj['sampled_smiles_woAAbackbone'][10:30]
    # target_smiles_woAAbackbone = [remove_AA_backbone_from_product_smiles(target_smile, aa_smiles=reactant.split('.')[0]) for target_smile,reactant in zip(target_smiles, reactants)]
    # following 3 are the same result (0.7964)
    # print(compute_substructure_accuracy(sampled_smiles, target_smiles_woAAbackbone, np.array([1,3])))
    # print(compute_substructure_accuracy(sampled_smiles_woAAbackbone, target_smiles_woAAbackbone, np.array([1,3])))
    print(compute_substructure_accuracy(sampled_smiles, target_smiles, np.array([1,3,5,10])))
    
    # following 2 are the same result (0.7168)
    print(compute_exact_match_accuracy(sampled_smiles, target_smiles, np.array([1,3,5,10])))
    # print(compute_exact_match_accuracy(sampled_smiles_woAAbackbone, target_smiles_woAAbackbone, np.array([1,3])))
    pdb.set_trace()
    # sampled_smiles_woAAbackbone = [[remove_AA_backbone_from_product_smiles(smi) for smi in smiles_list] for smiles_list in sampled_smiles]
    acc, top_Ks = compute_ligand_acc(sampled_smiles_woAAbackbone, target_smiles_woAAbackbone, np.array([1,3,5,10]))
    targets_woAAbackbone = [remove_AA_backbone_from_product_smiles(product_smiles) for product_smiles in targets]
    acc = compute_ligand_acc(smiles,targets_woAAbackbone,np.array([1,3,5,10]))
    print(acc)