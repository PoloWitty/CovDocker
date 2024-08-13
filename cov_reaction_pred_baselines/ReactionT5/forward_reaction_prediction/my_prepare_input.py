"""
desc:	prepare input data for ReactionT5
author:	Yangzhe Peng
date:	2024/05/12
"""


import pandas as pd
import tqdm


if __name__=='__main__':
    
    only_unseen = True
    
    if only_unseen:
        dataset_filename = '../../../data/processed/dataset.unseen.csv'
    else:
        dataset_filename = '../../../data/processed/dataset.csv'
    
    df = pd.read_csv(dataset_filename)
    
    splits = ['train','test'] if not only_unseen else ['unseen']
    for set_ in splits:
        split_df = df[df['set']==set_]
        input_df_list = []
        for i,row in tqdm.tqdm(split_df.iterrows(),total=len(split_df),desc='preparing input data'):
            reactant = row['reactants']
            product = row['products']
            input_df_list.append({
                'REACTANT': reactant,
                'PRODUCT': product,
                'CATALYST': ' ',
                'REAGENT':' ',
                'SOLVENT':' ',
                'input': f'REACTANT:{reactant}REAGENT: '
            })
        input_df_list = pd.DataFrame(input_df_list)
        input_df_list.to_csv(f'./data/input4ReactionT5.{set_}.csv',index=False)
    