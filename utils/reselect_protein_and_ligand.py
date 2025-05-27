
import pandas as pd
import os
import tqdm

import multiprocessing
import subprocess
from prody import parsePDB, writePDB

def run(cmd):
    subprocess.run(cmd, shell=True, check=True, timeout=60*15)

def parse(x):
    i,row = x
    ligand_pdb_filename = row['bonded ligand']
    protein_pdb_filename = row['bonded protein'] # ./data/processed/bonded/4WSJ/4WSJ_protein.pdb
    pdb_id = protein_pdb_filename.split('/')[-2]
    bond_pocket_atom, bond_pocket_res, bond_pocket_chain,bond_pocket_chain_idx = row['bond'].split('-')[0].split(' ') # OD2 ASP D 229-CAF 3U3 D 501
    bond_ligand_atom, bond_ligand_het,bond_ligand_chain,bond_ligand_chain_idx = row['bond'].split('-')[1].split(' ')
    
    old_ligand_pdb_filename = ligand_pdb_filename.replace(".pdb",".old.pdb")
    old_protein_pdb_filename = protein_pdb_filename.replace(".pdb",".old.pdb")

    # rename old files
    if not os.path.exists(old_ligand_pdb_filename):
        run(f'mv {ligand_pdb_filename} {old_ligand_pdb_filename}')
    if not os.path.exists(old_protein_pdb_filename):
        run(f'mv {protein_pdb_filename} {old_protein_pdb_filename}')
    
    # select ligand
    ligand_pdb = parsePDB(old_ligand_pdb_filename)
    if ligand_pdb == None:
        return pdb_id
    sele = ligand_pdb.select(f'resnum {bond_ligand_chain_idx} and chain {bond_ligand_chain} and resname {bond_ligand_het} and altloc _')
    if sele!= None:
        writePDB(ligand_pdb_filename, sele)
    else:
        return pdb_id
    
    # select protein
    protein_pdb = parsePDB(old_protein_pdb_filename)
    check_sele = protein_pdb.select(f'resnum {bond_pocket_chain_idx} and chain {bond_pocket_chain} and resname {bond_pocket_res} and not altloc _')

    # If the selection returns any atoms, it means there are alternate locations at bond amino acid
    if check_sele:
        return pdb_id
    else:
        # If no alternate locations, select all atoms in the chain where altloc is " " or "A"
        selection_str = f'altloc _ A'
        sele = protein_pdb.select(selection_str)
        if sele!=None:
            writePDB(protein_pdb_filename, sele)
        else:
            return pdb_id
    return True

if __name__=='__main__':
    dataset_info_filename = './data/processed/dataset.csv'
    df = pd.read_csv(dataset_info_filename)

    failed_pdb = []
    with tqdm.trange(len(df), desc=f're-selecting') as pbar:
        with multiprocessing.Pool(20) as pool:
            for res in pool.imap(parse, df.iterrows()):
                if res != True:
                    failed_pdb.append(res)
                pbar.update()
    print(f"failed pdb {len(failed_pdb)}\n {failed_pdb}")
# failed pdb 248
#  ['6Q35', '6H5C', '5WKL', '1PWG', '2O7S', '6QWC', '2Y62', '6PGP', '3ESC', '5WKM', '1HL9', '4FI7', '4XTI', '6DI5', '2CE2', '5TNE', '2Y61', '6PNM', '5EKY', '6QWA', '5EMY', '3PAA', '5TNK', '2QNY', '4OE7', '6PUJ', '5URC', '2CL6', '5TKL', '4X2P', '1KVL', '2CLC', '6PNO', '5LNS', '5UJO', '1HVB', '5LWN', '4UQL', '6BID', '5U4F', '6MZ2', '6QW8', '2ZD8', '4KKY', '5ZWH', '2AAZ', '3CMC', '1H2J', '5TND', '1NY0', '4CLM', '5V6S', '3O88', '4PNC', '5WKJ', '5O4A', '2XGI', '2FFY', '6SFE', '2XU3', '4PA8', '5P9L', '6QW9', '6AFB', '6V7H', '4A2R', '6RWS', '6O9W', '1YMX', '3KNE', '4FI8', '6SHI', '6FFM', '6B95', '5W2Y', '6TD1', '6GXY', '5ZA2', '1BWC', '4H8R', '1LLB', '6H2H', '1Q2Q', '1H7N', '3PBT', '1BLC', '1PWC', '5W2U', '3SZB', '4I9B', '2QNZ', '6CWY', '4NMM', '6N9P', '5TNJ', '6G3Z', '4XTD', '3W09', '6QWB', '2EVW', '2XQT', '3LCE', '4MZ4', '6DQB', '6PUE', '6B0Y', '4UPV', '6H5D', '3HJ0', '1YLV', '1YQS', '1TDG', '1LPB', '6DI1', '4L1S', '1LL9', '4WKT', '3O87', '6TD0', '5U4G', '1PWD', '3WBE', '1PW8', '1W31', '1MPL', '1YM1', '1KEQ', '2CL7', '6P4E', '1GJP', '3SZ9', '6SJU', '5DGJ', '1P11', '5EZS', '3PA9', '3O86', '4FI6', '3TNO', '2YJ0', '5V9U', '6SHH', '4UQP', '4HJT', '1QX1', '6DI9', '4WEF', '6V7I', '6H5G', '4WEG', '5WKK', '5E83', '3S19', '5HZQ', '4HJS', '2WKH', '1PI4', '1M33', '6BQ0', '5F4S', '5W2W', '5W26', '1H7O', '4CG2', '6SKB', '6SFG', '5HF8', '5HFA', '5HF9', '6HQV', '3BG8', '6T5Y', '2A4T', '5A2D', '3A2G', '6IEO', '6TPM', '6FFS', '2NLR', '5Y57', '2WVP', '6AF5', '6AFC', '6AFL', '1H11', '6U0Q', '1X84', '3KKU', '4KIO', '4XUZ', '4XUX', '5HWN', '4RQX', '5ZWE', '5TKC', '5TK3', '1GW1', '7B4E', '7B46', '7NTV', '7AWS', '7NJB', '7K0G', '7B4D', '7NL5', '6WYF', '7O59', '7NIZ', '7B48', '7LKW', '7LKX', '6ZNC', '6YNQ', '7ONK', '7LKS', '6Z25', '7JX1', '7NZV', '7LKU', '7LKR', '7LKV', '7LKT', '7APF', '7NR7', '7CB7', '7O6F', '6ZAM', '7ONN', '6YCG', '6Z23', '7NLA', '7NYE', '6XCC', '7O57', '7B49', '7B4C', '7B4B', '7B47', '7B4N', '7O3Q', '6WVO', '7AQJ', '7NM9', '7NUK', '7NSV', '7ORE', '6VIM', '6W2A']