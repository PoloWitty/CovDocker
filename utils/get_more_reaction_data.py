
import pdb
import tqdm
from rdkit import Chem
import pandas as pd
import pickle

AA2smiles = {'ALA': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])[H]', 'ARG': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])C([H])([H])C([H])([H])N([H])C(N([H])[H])=[N+]([H])[H]', 'ASN': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])C(=O)N([H])[H]', 'ASP': '[H]OC(=O)C([H])([H])[C@@]([H])(C(=O)O[H])N([H])[H]', 'CYS': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])S[H]', 'GLU': '[H]OC(=O)C([H])([H])C([H])([H])[C@@]([H])(C(=O)O[H])N([H])[H]', 'GLY': '[H]OC(=O)C([H])([H])N([H])[H]', 'HIS': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])c1nc([H])n([H])c1[H]', 'ILE': '[H]OC(=O)[C@@]([H])(N([H])[H])[C@@]([H])(C([H])([H])[H])C([H])([H])C([H])([H])[H]', 'LEU': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H]', 'LYS': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])C([H])([H])C([H])([H])C([H])([H])N([H])[H]', 'MET': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])C([H])([H])SC([H])([H])[H]', 'PRO': '[H]OC(=O)[C@@]1([H])N([H])C([H])([H])C([H])([H])C1([H])[H]', 'SER': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])O[H]', 'THR': '[H]OC(=O)[C@@]([H])(N([H])[H])[C@]([H])(O[H])C([H])([H])[H]', 'TYR': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])c1c([H])c([H])c(O[H])c([H])c1[H]', 'VAL': '[H]OC(=O)[C@@]([H])(N([H])[H])C([H])(C([H])([H])[H])C([H])([H])[H]'}
smiles2AA = { Chem.CanonSmiles(Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(v)))): k for k, v in AA2smiles.items()}

if __name__=='__main__':
    dataset_filename = 'uspto_sep.pickle'
    print(f'reading {dataset_filename}...')
    dataset = pickle.load(open(dataset_filename, 'rb'))
    
    AA_reaction = []
    for idx, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset),desc=f'processing {dataset_filename}'):
        reactants = row['reactants_mol']
        products = row['products_mol']
        reactants = Chem.CanonSmiles(Chem.MolToSmiles(reactants))
        products = Chem.CanonSmiles(Chem.MolToSmiles(products))
        reactants_list = reactants.split('.')
        if len(reactants_list) < 2 :
            continue
        for reactant in reactants_list:
            if smiles2AA.get(reactant,"") == "":
                AA_reaction.append({'reactants': reactants, 'products': products})
    
    pickle.dump(AA_reaction, open('uspto_sep.AA_subset.pickle', 'wb'))
    