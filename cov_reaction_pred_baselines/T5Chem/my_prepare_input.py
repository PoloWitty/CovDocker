"""
desc:	prepare input data for T5Chem
author:	Yangzhe Peng
date:	2024/05/12
"""


import pandas as pd
import tqdm


if __name__=='__main__':
    only_unseen = True
    
    if only_unseen:
        dataset_filename = '../../data/processed/dataset.unseen.csv'
    else:
        dataset_filename = '../../data/processed/dataset.csv'
        
    df = pd.read_csv(dataset_filename)
    
    splits = ['train','test'] if not only_unseen else ['unseen']
    for set_ in splits:
        split_df = df[df['set']==set_]
        input_source_list = []
        input_target_list = []
        for i,row in tqdm.tqdm(split_df.iterrows(),total=len(split_df),desc='preparing input data'):
            reactant = row['reactants']
            product = row['products']
            input_source_list.append(reactant)
            input_target_list.append(product)
        
        with open(f'./data/covDocker/{set_}.source', 'a') as fp:
            fp.truncate(0)
            for source in input_source_list:
                fp.write(source + '\n')
        
        
        with open(f'./data/covDocker/{set_}.target', 'a') as fp:
            fp.truncate(0)
            for target in input_target_list:
                fp.write(target + '\n')
        