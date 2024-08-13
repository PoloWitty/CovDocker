"""
desc:   use streamlit to viz reaction prediction results and draw chemical reaction
author:	Yangzhe Peng
date:	2024/04/04
"""


import streamlit as st
from streamlit_extras.row import row as Row
import os
import pdb
from math import ceil
import pandas as pd
import tqdm
from functools import lru_cache
import base64

from IPython.display import SVG, display


from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Descriptors import MolWt

batch_size = 15
img_dir = './tmp/'
statistics_filename = './data/statistic/dataset.statistics.csv'
# confidence_filename = './Chemformer/tmp/confidence_data.csv'
# confidence_pred_filename = './Chemformer/tmp/confidence_dist_pred.csv'
os.makedirs(img_dir, exist_ok=True)


def draw_2dmol(mol, filename='tmp.png'):
    if type(mol) != str: # None or NaN
        return 
    mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    Draw.MolToFile(mol,filename,size=(600,600))

def draw_chemical_reaction(smiles, highlightByReactant=False, font_scale=1.5):
    rxn = rdChemReactions.ReactionFromSmarts(smiles,useSmiles=True)
    trxn = rdChemReactions.ChemicalReaction(rxn)
    d2d = rdMolDraw2D.MolDraw2DSVG(800,300)
    d2d.drawOptions().annotationFontScale=font_scale
    d2d.DrawReaction(trxn,highlightByReactant=highlightByReactant)

    d2d.FinishDrawing()

    return d2d.GetDrawingText()

def initialize():
    # rxn pred data
    dataset_df = pd.read_csv(statistics_filename)
    for i,row in tqdm.tqdm(dataset_df.iterrows(), total = len(dataset_df)):
        if os.path.exists(f'{img_dir}/{i}_reactants.png'):
            continue
        draw_2dmol(row['reactants'], f'{img_dir}/{i}_reactants.png')
        draw_2dmol(row['products'], f'{img_dir}/{i}_products.png')
        draw_2dmol(row['pred_products'], f'{img_dir}/{i}_pred_products.png')
        
    df = dataset_df[['reactants','products','pred_products','substructure_res','exact_match_res']]
    
    # # confidence data
    # confidence_df = pd.read_csv(confidence_filename)
    # confidence_data = []
    # for i in tqdm.tqdm(range(len(confidence_df)), total = len(confidence_df)):
    #     if i %2 == 1:
    #         continue
    #     row = confidence_df.iloc[i]
    #     reactants = row['rxn'].split('>>')[0]
    #     products = row['rxn'].split('>>')[1]
    #     row_plus1 = confidence_df.iloc[i+1]
    #     reactants_plus1 = row_plus1['rxn'].split('>>')[0]
    #     products_plus1= row_plus1['rxn'].split('>>')[1]
    #     assert reactants == reactants_plus1
    #     confidence_data.append({
    #         'reactants': reactants,
    #         'pos_products': products,
    #         'neg_products': products_plus1,
    #         'idx':i
    #     })
    #     if os.path.exists(f'{img_dir}/confidence_{i}_reactants.png'):
    #         continue
    #     draw_2dmol(reactants, f'{img_dir}/confidence_{i}_reactants.png')
    #     draw_2dmol(products, f'{img_dir}/confidence_{i}_pos_products.png')
    #     draw_2dmol(products_plus1, f'{img_dir}/confidence_{i}_neg_products.png')
    # confidence_df = pd.DataFrame(confidence_data).set_index('idx')

    # # confidence pred data
    # confidence_pred_df = pd.read_csv(confidence_pred_filename)
    # confidence_pred_data = []
    # for i in tqdm.tqdm(range(len(confidence_pred_df)), total = len(confidence_pred_df)):
    #     if i %2 == 1:
    #         continue
    #     row = confidence_pred_df.iloc[i]
    #     reactants = row['rxn'].split('>>')[0]
    #     products = row['rxn'].split('>>')[1]
    #     pos_dist = row['pred_dist']
    #     row_plus1 = confidence_pred_df.iloc[i+1]
    #     reactants_plus1 = row_plus1['rxn'].split('>>')[0]
    #     products_plus1= row_plus1['rxn'].split('>>')[1]
    #     neg_dist = row_plus1['pred_dist']
    #     assert reactants == reactants_plus1
    #     confidence_pred_data.append({
    #         'reactants': reactants,
    #         'pos_products': products,
    #         'pos_dist': pos_dist,
    #         'neg_products': products_plus1,
    #         'neg_dist': neg_dist,
    #         'idx':i
    #     })
    #     if os.path.exists(f'{img_dir}/confidence_pred_{i}_reactants.png'):
    #         continue
    #     draw_2dmol(reactants, f'{img_dir}/confidence_pred_{i}_reactants.png')
    #     draw_2dmol(products, f'{img_dir}/confidence_pred_{i}_pos_products.png')
    #     draw_2dmol(products_plus1, f'{img_dir}/confidence_pred_{i}_neg_products.png')
    # confidence_pred_df = pd.DataFrame(confidence_pred_data).set_index('idx')
    confidence_df = None; confidence_pred_df = None
    return df,confidence_pred_df,confidence_pred_df

if 'confidence_pred_df' not in st.session_state:
    df,confidence_df,confidence_pred_df = initialize()
    st.session_state.df = df
    st.session_state.confidence_df = confidence_df
    st.session_state.confidence_pred_df = confidence_pred_df
else:
    df = st.session_state.df 
    confidence_df = st.session_state.confidence_df
    confidence_pred_df = st.session_state.confidence_pred_df

tab1, tab2, tab3, tab4 = st.tabs(["Data Viz", "Draw reaction", 'Confidence Data Viz', 'Confidence Pred Viz'])

with tab1:
    controls = st.columns(2)
    with controls[0]:
        substructure_res = st.selectbox("Substructure res:", ['True','False','None'])
    with controls[1]:
        exact_match_res = st.selectbox("Exact match res:", ['True','False','None'])

    grid = st.columns(3)

    if substructure_res != 'None':
        substructure_res = True if substructure_res == 'True' else False
        subset_df = df[ df['substructure_res']==substructure_res ]
    else:
        subset_df = df
    if exact_match_res != 'None':
        exact_match_res = True if exact_match_res == 'True' else False
        subset_df = subset_df[ subset_df['exact_match_res']==exact_match_res ]
    else:
        subset_df = subset_df

    calc_molw = lambda smiles: MolWt(Chem.MolFromSmiles(smiles))

    with grid[0]:
        st.write('### Reactants')
    with grid[1]:
        st.write('### Products')
    with grid[2]:
        st.write('### Pred Products')
    for idx,row in subset_df.iterrows():
        vis_row = Row(3)
        for col in ['reactants','products','pred_products']:
            if os.path.exists(f'{img_dir}/{idx}_{col}.png'):
                smiles = subset_df.at[idx,f'{col}']
                # vis_row.image(f'{img_dir}/{idx}_{col}.png', caption=f"{idx}:[{calc_molw(smiles):.2f}]{smiles}", width=700 , use_column_width='always')
                vis_row.image(f'{img_dir}/{idx}_{col}.png', caption=f"{idx}:{smiles}", width=700 , use_column_width='always')
            else:
                vis_row.write('##')
                vis_row.write('##')
                vis_row.write(f"{idx}:{subset_df.at[idx,f'{col}']}")

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

with tab2:
    reaction = st.text_input('Reaction:', 'CCO.CC(=O)O>>CC(=O)O.CCO')
    svg = draw_chemical_reaction(reaction,  highlightByReactant=True, font_scale=0.5)
    render_svg(svg)
    
# with tab3:
#     num_batches = ceil(len(confidence_df)/batch_size)
#     page = st.selectbox('Page', range(1, num_batches+1))
#     batch = confidence_df.iloc[(page-1)*batch_size: page*batch_size]
#     grid_tab3 = st.columns(3)
#     with grid_tab3[0]:
#         st.write('### Reactants')
#     with grid_tab3[1]:
#         st.write('### Pos Products')
#     with grid_tab3[2]:
#         st.write('### Neg Products')
#     for idx,row in batch.iterrows():
#         vis_row = Row(3)
#         for col in ['reactants','pos_products','neg_products']:
#             if os.path.exists(f'{img_dir}/confidence_{idx}_{col}.png'):
#                 smiles = confidence_df.at[idx,f'{col}']
#                 vis_row.image(f'{img_dir}/confidence_{idx}_{col}.png', caption=f"{idx}:{smiles}", width=700 , use_column_width='always')
#             else:
#                 vis_row.write('##')
#                 vis_row.write('##')
#                 vis_row.write(f"{idx}:{confidence_df.at[idx,f'{col}']}")

# with tab4:
#     acc = (confidence_pred_df['pos_dist'] < confidence_pred_df['neg_dist']).sum() / len(confidence_pred_df)
#     print(acc)
    
#     num_batches = ceil(len(confidence_pred_df)/batch_size)
#     page = st.selectbox('Page', range(1, num_batches+1))
#     batch = confidence_pred_df.iloc[(page-1)*batch_size: page*batch_size]
#     grid_tab3 = st.columns(3)
#     with grid_tab3[0]:
#         st.write('### Reactants')
#     with grid_tab3[1]:
#         st.write('### Pos Products')
#     with grid_tab3[2]:
#         st.write('### Neg Products')
#     for idx,row in batch.iterrows():
#         vis_row = Row(3)
#         for col in ['reactants','pos_products','neg_products']:
#             if os.path.exists(f'{img_dir}/confidence_pred_{idx}_{col}.png'):
#                 smiles = confidence_pred_df.at[idx,f'{col}']
#                 pre_ = col.split('_')
#                 if len(pre_) == 2:
#                     pre_ = pre_[0]
#                     dist = confidence_pred_df.at[idx,f'{pre_}_dist']
#                 else:
#                     dist = 0
#                 vis_row.image(f'{img_dir}/confidence_pred_{idx}_{col}.png', caption=f"{idx}[{dist}]:{smiles}", width=700 , use_column_width='always')
#             else:
#                 vis_row.write('##')
#                 vis_row.write('##')
#                 vis_row.write(f"{idx}:{confidence_pred_df.at[idx,f'{col}']}")

