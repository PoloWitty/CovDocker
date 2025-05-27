"""
desc:	split the complex pdb file into ground truth protein and ligand files.
author:	Yangzhe Peng
date:	2023/12/24
"""


import pdb
import re
import os
import tqdm
import argparse
from collections import defaultdict

import pandas as pd

from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select

from rdkit import Chem


# only consider the 20 standard amino acids
# three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                # 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                # 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


three_to_one = {'ALA':	'A',
'ARG':	'R',
'ASN':	'N',
'ASP':	'D',
'CYS':	'C',
'GLN':	'Q',
'GLU':	'E',
'GLY':	'G',
'HIS':	'H',
'ILE':	'I',
'LEU':	'L',
'LYS':	'K',
'MET':	'M',
'MSE':  'M', # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
'PHE':	'F',
'PRO':	'P',
'PYL':	'O',
'SER':	'S',
'SEC':	'U',
'THR':	'T',
'TRP':	'W',
'TYR':	'Y',
'VAL':	'V',
'ASX':	'B',
'GLX':	'Z',
'XAA':	'X',
'XLE':	'J'}

class SelectProtein(Select):
    def accept_residue(self, residue):
        hetero, resid, insertion = residue.full_id[-1]
        return hetero == ' ' and residue.get_resname() in three_to_one 
        #  hetero flag, “W” for waters, “H” for hetero residues, otherwise blank.

class SelectLigand(Select):
    def __init__(self,het_id,chain_id) -> None:
        super().__init__()
        self.het_id = het_id
        self.chain_id = chain_id
    def accept_residue(self, residue):
        hetero, resid, insertion = residue.full_id[-1]
        return residue.get_resname() == self.het_id and residue.get_parent().get_id() == self.chain_id

def get_smiles_from_sdf(sdf_file, sanitize=True):
    supplier = Chem.SDMolSupplier(sdf_file, sanitize=sanitize)
    smiles_list = [Chem.MolToSmiles(mol) for mol in supplier if mol is not None]
    if len(smiles_list) > 1:
        print(f'Warning: more than one mol in ligand sdf file ({sdf_file})')
    elif len(smiles_list) == 0:
        print(f'Warning: no mol in ligand sdf file ({sdf_file})')
        return None
    return smiles_list[0]

def extract_formul_fields(formul_line):
    '''
    write according to https://www.wwpdb.org/documentation/file-format-content/format33/sect4.html#FORMUL
    '''
    record_name = formul_line[0:6].strip()
    comp_num = int(formul_line[8:10].strip())
    het_id = formul_line[12:15].strip()
    continuation = int(formul_line[16:18].strip()) if formul_line[16:18].strip() else None
    asterisk = formul_line[18].strip() if formul_line[18].strip() else None
    text = formul_line[19:70].strip()

    return record_name, comp_num, het_id, continuation, asterisk, text

def extract_modres_fields(modres_line):
    '''
    write according to https://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#MODRES
    '''
    record_name = modres_line[0:6].strip()
    id_code = modres_line[7:11].strip()
    res_name = modres_line[12:15].strip()
    chain_id = modres_line[16].strip()
    seq_num = int(modres_line[18:22].strip())
    i_code = modres_line[22].strip() if modres_line[22].strip() else None
    std_res = modres_line[24:27].strip()
    comment = modres_line[29:70].strip()

    return record_name, id_code, res_name, chain_id, seq_num, i_code, std_res, comment

def extract_link_fields(link_line):
    '''
    write according to https://www.wwpdb.org/documentation/file-format-content/format33/sect6.html#LINK
    '''
    record_name = link_line[0:6].strip()
    name1 = link_line[12:16].strip()
    alt_loc1 = link_line[16].strip() if link_line[16].strip() else None
    res_name1 = link_line[17:20].strip()
    chain_id1 = link_line[21].strip() if link_line[21].strip() else None
    res_seq1 = int(link_line[22:26].strip())
    i_code1 = link_line[26].strip() if link_line[26].strip() else None
    name2 = link_line[42:46].strip()
    alt_loc2 = link_line[46].strip() if link_line[46].strip() else None
    res_name2 = link_line[47:50].strip()
    chain_id2 = link_line[51].strip() if link_line[51].strip() else None
    res_seq2 = int(link_line[52:56].strip())
    i_code2 = link_line[56].strip() if link_line[56].strip() else None
    sym1 = link_line[59:65].strip()
    sym2 = link_line[66:72].strip()
    length = float(link_line[73:78].strip())

    return record_name, name1, alt_loc1, res_name1, chain_id1, res_seq1, i_code1, name2, alt_loc2, res_name2, chain_id2, res_seq2, i_code2, sym1, sym2, length

def read_pdb2het_from_file(path:str):
    pdb2het = {}
    with open(path, 'r') as fp:
        for line in fp:
            pdb,het = line.strip().split(',')
            pdb2het[pdb] = het
    return pdb2het

def found_data_and_covalent_bond_from_pdb_file(complex_file:str, het_id:str):
    # found the covalent bound information from pdb file (PDBparser can not parse this information)
    raw_bonds = []
    raw_formuls = []
    raw_modres = []
    date = ''
    with open(complex_file, 'r') as tmp_fp:
        for line in tmp_fp:
            if line.startswith('HEADER'):
                date = line[50:59].strip()
            if line.startswith("LINK"):
                raw_bonds.append(line.strip('\n'))
            if line.startswith("FORMUL"):
                raw_formuls.append(line.strip('\n'))
            if line.startswith("MODRES"):
                raw_modres.append(line.strip('\n'))

    filtered_bond_to_idx = {}
    bonds = {'p-l': [], 'length': []}
    for bond in raw_bonds:
        # example link line:
        # LINK         B   T29 H   1                 OG  SER H 195     1555   1555  1.60
        try:
            _, name1, _, res_name1, chain_id1, res_seq1, _, name2, _, res_name2, chain_id2, res_seq2, _, _, _, length = extract_link_fields(bond)
        except ValueError:
            print(f'{bond} can not be parsed, and will be skipped')
            info['bond_has_no_length'].append(complex+':    '+bond)
            continue

        if res_name1 == het_id and res_name2 in three_to_one: # the first one is the ligand
            bonds['p-l'].append(f'{name2} {res_name2} {chain_id2} {res_seq2}-{name1} {res_name1} {chain_id1} {res_seq1}')
            new_bond = f'{name2} {res_name2}-{name1} {res_name1}'
        elif res_name2 == het_id and res_name1 in three_to_one: # the second one is the ligand
            bonds['p-l'].append(f'{name1} {res_name1} {chain_id1} {res_seq1}-{name2} {res_name2} {chain_id2} {res_seq2}')
            new_bond = f'{name1} {res_name1}-{name2} {res_name2}'
        else:
            continue                
        bonds['length'].append(str(length))
        filtered_bond_to_idx[new_bond] = len(bonds['p-l'])-1

    final_bonds = {'p-l': [], 'length': []}
    for idx in filtered_bond_to_idx.values():
        final_bonds['p-l'].append(bonds['p-l'][idx])
        final_bonds['length'].append(bonds['length'][idx])
    return date, final_bonds

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='split the complex pdb file into ground truth protein and ligand files.')
    # input
    parser.add_argument('--covpdb_complexes_path', type=str, default='./data/covpdb/CovPDB_complexes/',
                        help='path to the CovPDB complexes (input)')
    parser.add_argument('--covbinderInPDB_complexes_path', type=str, default='./data/covbinderInPDB/pdb/',
                        help='path to the covbinderInPDB complexes (input)')
    parser.add_argument('--covbinderInPDB_info_filename', type=str, default='./data/covbinderInPDB/CovBinderInPDB_2022Q4_AllRecords.csv',
                        help='path to the covbinderInPDB info file (input)')
    # output
    parser.add_argument("--save_dir", type=str, default="./data/processed/bonded",
                        help="Specify where to save the processed files (output)")
    parser.add_argument('--dataset_info_filename', type=str, default='./data/processed/examples.csv',
                        help='path to the index file (output)')
    # other settings
    parser.add_argument('--drop_more_than_one_bond', action='store_true',default=True,
                        help='drop the complexes that have more than one covalent bond between ligand and protein')
    args = parser.parse_args()

    covpdb_complexes_path = args.covpdb_complexes_path

    # some preparation work
    os.makedirs(args.save_dir, exist_ok=True)
    pdb_parser = PDBParser(QUIET=True)
    pdb_io = PDBIO()
    protein_residues = SelectProtein()   
    info = defaultdict(list) 
    pdb2het = read_pdb2het_from_file('./data/processed/pdb2het.csv')
    covbinderInPDB_info = pd.read_csv(args.covbinderInPDB_info_filename)

    with open(args.dataset_info_filename, 'w') as fp:
        # write the header
        fp.write('date,src,pre-reactive smiles,bonded protein,bonded ligand,bond,length\n')
        for complex in tqdm.tqdm(os.listdir(covpdb_complexes_path), desc='[1] processing covpdb complexes'):
            # get the complex filename
            complex_file = os.path.join(covpdb_complexes_path, complex, f'{complex}.pdb')
            # get the pre-reactive ligand filename
            ligand_file_name = os.listdir(os.path.join(covpdb_complexes_path, complex))[1]
            if not ligand_file_name.endswith('.sdf'):
                ligand_file_name = os.listdir(os.path.join(covpdb_complexes_path, complex))[0]
            ligand_file = os.path.join(covpdb_complexes_path, complex, ligand_file_name)

            complex = complex.upper() # pdb_id in covpdb are lower cased

            # get smiles
            smiles = get_smiles_from_sdf(ligand_file)
            if smiles is None:
                print(f'{complex} is skipped because can not get the smiles for the pre-reactive ligand')
                info['no_smiles'].append(complex)
                continue

            
            # get date and bond
            date, bonds = found_data_and_covalent_bond_from_pdb_file(complex_file, pdb2het[complex.upper()])
            if date == '':
                info['no_date'].append(complex)
                continue
            if len(bonds['p-l']) == 0:
                info['no_available_link'].append(complex)
                continue
            elif len(bonds['p-l']) > 1:
                info['more_than_one_link'].append(complex)
                if args.drop_more_than_one_bond:
                    continue

            os.makedirs(f'{args.save_dir}/{complex}', exist_ok=True)
            structure = pdb_parser.get_structure(complex, complex_file)
            pdb_io.set_structure(structure)
            # get protein structure
            protein_path = f'{args.save_dir}/{complex}/{complex}_protein.pdb'
            pdb_io.save(protein_path, protein_residues)
            # extract ligand structure
            ligand_path = f'{args.save_dir}/{complex}/{complex}_ligand.pdb'
            chain_id = bonds['p-l'][0].split('-')[1].split(' ')[2] # {name1} {res_name1} {chain_id1} {res_seq1}-{name2} {res_name2} {chain_id2} {res_seq2}
            ligand_residues = SelectLigand(pdb2het[complex.upper()],chain_id)
            pdb_io.save(ligand_path, ligand_residues)

            fp.write(f'{date},covpdb,{smiles},{protein_path},{ligand_path},{";".join(bonds["p-l"])},{";".join(bonds["length"])}\n')

        for index, row in tqdm.tqdm(covbinderInPDB_info.iterrows(),total=len(covbinderInPDB_info),desc='[2] processing covbinderInPDB complexes'):
            pdb_id = row['pdb_id']
            het_id = row['binder_id_in_adduct']

            pdb_id = pdb_id.upper()

            # only keep the covpdb result if the pdb file is also in covpdb
            # and only keep the first bond if there are more than one bond
            if os.path.exists(f'{args.save_dir}/{pdb_id}'):
                continue

            # get the complex filename
            complex_file = os.path.join(args.covbinderInPDB_complexes_path, f'{pdb_id}.pdb')

            # get smiles
            smiles = row['binder_smiles']
            try:
                mol = Chem.MolFromSmiles(smiles)
                smiles = Chem.MolToSmiles(mol)
            except:
                if pdb_id not in info['non_standard_smiles']:
                    info['non_standard_smiles'].append(pdb_id)
                continue

            # get date and bond
            date, bonds = found_data_and_covalent_bond_from_pdb_file(complex_file, het_id)
            if date == '':
                if pdb_id not in info['no_date']:
                    info['no_date'].append(pdb_id)
                continue
            if len(bonds['p-l']) == 0:
                if pdb_id not in info['no_available_link']:
                    info['no_available_link'].append(pdb_id)
                continue
            elif len(bonds['p-l']) > 1:
                if pdb_id not in info['more_than_one_link']:
                    info['more_than_one_link'].append(pdb_id)
                if args.drop_more_than_one_bond:
                    continue


            os.makedirs(f'{args.save_dir}/{pdb_id}', exist_ok=True)
            structure = pdb_parser.get_structure(pdb_id, complex_file)
            pdb_io.set_structure(structure)
            # get protein structure
            protein_path = f'{args.save_dir}/{pdb_id}/{pdb_id}_protein.pdb'
            pdb_io.save(protein_path, protein_residues)
            # extract ligand structure
            ligand_path = f'{args.save_dir}/{pdb_id}/{pdb_id}_ligand.pdb'
            chain_id = bonds['p-l'][0].split('-')[1].split(' ')[2] # {name1} {res_name1} {chain_id1} {res_seq1}-{name2} {res_name2} {chain_id2} {res_seq2}
            ligand_residues = SelectLigand(het_id,chain_id)
            pdb_io.save(ligand_path, ligand_residues)

            fp.write(f'{date},covbinderInPDB,{smiles},{protein_path},{ligand_path},{";".join(bonds["p-l"])},{";".join(bonds["length"])}\n')

    print('\n\n**********info**********')
    for k,v in info.items():
        print(f'{k}: {len(v)}')
        print(v)
    print('**********info**********')