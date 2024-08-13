"""
desc:	use covbinderInPDB_pdb2mechanism and covpdb_pdb2mechanism to get the final pdb2mechanism
author:	Yangzhe Peng
date:	2024/08/03
"""

import pandas as pd
import tqdm

if __name__=='__main__':
    # read the csv file
    covpdb_pdb2mechanism = pd.read_csv('covpdb_pdb2mechanism.csv')
    covbinderInPDB_pdb2mechanism = pd.read_csv('covbinderInPDB_pdb2mechanism.csv')

    # read data info file
    df = pd.read_csv('./data/processed/dataset.csv')
    pdb2mechanism = {}
    
    # Iterate over each row in df
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        # Get the pdb_id and src from the current row
        pdb_id = row['pdb_id']
        src = row['src']
        
        # Use covbinderInPDB_pdb2mechanism if src is covbinderInPDB
        if src == 'covbinderInPDB':
            mechanism = covbinderInPDB_pdb2mechanism.loc[covbinderInPDB_pdb2mechanism['pdb_id'] == pdb_id, 'mechanism'].values[0]
        else:
            mechanism = covpdb_pdb2mechanism.loc[covpdb_pdb2mechanism['pdb_id'] == pdb_id, 'mechanism'].values[0]
        
        pdb2mechanism[pdb_id] = mechanism
    
    # Save the pdb2mechanism dictionary to a csv file
    pdb2mechanism_df = pd.DataFrame(pdb2mechanism.items(), columns=['pdb_id', 'mechanism'])
    pdb2mechanism_df.to_csv('pdb2mechanism.csv', index=False)
    