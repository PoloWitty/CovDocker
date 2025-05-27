import os
import pdb
import pickle
import argparse
import random
from typing import List, Dict, Any

import tqdm
import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

import numpy as np
import torch
from scipy.spatial.distance import cdist
from rdkit import Chem

from unimol_tools import UniMolRepr

from molbart.modules.atom_assign import compute_substructure_accuracy, compute_exact_match_accuracy
from molbart.modules.data.util import BatchEncoder
from molbart.modules.tokenizer import ChemformerTokenizer
from molbart.models.confidence_model import BERTModel

BEAM_SIZE = 50

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[ INFO ] Random seed: \t [{seed}]")

class ConfidencePredictor:
    def __init__(self, model_path, vocabulary_path, max_seq_len):
        self.tokenizer = ChemformerTokenizer(filename=vocabulary_path)
        self.batch_encoder = BatchEncoder(
            tokenizer=self.tokenizer, masker=None, max_seq_len=max_seq_len
        )
        self.model = BERTModel.load_from_checkpoint(model_path)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
    
    def _transform_input(self, reactants:List[str], products:List[str]):
        reactants_ids, reactants_mask = self.batch_encoder(
            reactants, add_sep_token=False
        )
        products_ids, products_mask = self.batch_encoder(
            products, add_sep_token=False
        )
        return reactants_ids, reactants_mask, products_ids, products_mask
    
    def on_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move data in "batch" to the current model device.

        Args:
            batch (Dict[str, Any]): batch input data to model.
        Returns:
            Dict[str, Any]: batch data on current device.
        """
        device_batch = {
            key: val.to(self.device) if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()
        }
        return device_batch
    
    def _calc_batch_dist(self, reactants_emb:torch.Tensor, products_emb:torch.Tensor):
        # return torch.cdist(reactants_emb, products_emb, p=2)[0]
        cos_sim = torch.nn.functional.cosine_similarity(reactants_emb, products_emb, dim=1).cpu().numpy()
        # assert (cos_sim >= 0).all() and (cos_sim <= 1).all(), f'cosine similarity should be in [0,1], but got {cos_sim}'
        return cos_sim
    
    def __call__(self,reactants:List[str], products_list:List[List[str]]):
        '''
        products_list[i] should be the predicted product list of reactants[i]
        '''
        sorted_products = []
        for reactant, products in tqdm.tqdm(zip(reactants, products_list), desc='predicting', total=len(reactants)):
            assert type(reactant)==str and type(products) == list
            reactant_ids, reactant_mask, products_ids, products_mask = self._transform_input([reactant], products)
            with torch.no_grad():
                model_input = {
                    'reactants_input': reactant_ids,
                    'reactants_pad_mask': reactant_mask,
                    'products_input': products_ids,
                    'products_pad_mask': products_mask
                }
                model_input = self.on_device(model_input)
                model_output = self.model(model_input)
                dist = self._calc_batch_dist(model_output['reactants_emb'].cpu(), model_output['products_emb'].cpu())
                sorted_indices = np.argsort( -dist ) # desc order
                sorted_product = np.array(products)[sorted_indices].tolist()
                sorted_products.append(sorted_product)
        return sorted_products

class UnimolPredictor:
    def __init__(self):
        self.encoder = UniMolRepr(data_type='molecule')
    
    def _calc_batch_dist(self, reactants_emb:torch.Tensor, products_emb:torch.Tensor):
        # return torch.cdist(reactants_emb, products_emb, p=2)[0]
        cos_sim = torch.nn.functional.cosine_similarity(reactants_emb, products_emb, dim=1).cpu().numpy()
        # assert (cos_sim >= 0).all() and (cos_sim <= 1).all(), f'cosine similarity should be in [0,1], but got {cos_sim}'
        return cos_sim
    
    def __call__(self,reactants:List[str], products_list:List[List[str]]):
        '''
        products_list[i] should be the predicted product list of reactants[i]
        '''
        sorted_products = []
        # # too slow version
        # for reactant, products in tqdm.tqdm(zip(reactants, products_list), desc='predicting', total=len(reactants)):
        #     assert type(reactant)==str and type(products) == list
        #     with torch.no_grad():
        #         reactant_emb = torch.tensor(self.encoder.get_repr([reactant])['cls_repr']).squeeze(0)
        #         products_ = [ product if Chem.MolFromSmiles(product)!=None else 'O' for product in products]
        #         products_emb = torch.tensor(self.encoder.get_repr(products_)['cls_repr'])
        #         dist = self._calc_batch_dist(reactant_emb, products_emb)
        #         sorted_indices = np.argsort( -dist ) # desc order
        #         sorted_product = np.array(products)[sorted_indices].tolist()
        #         sorted_products.append(sorted_product)
        
        # # faster version
        # reactants_emb = self.encoder.get_repr(reactants)['cls_repr']
        # products_ = [ product if Chem.MolFromSmiles(product)!=None else 'O' for products in products_list for product in products ]
        # products_emb = self.encoder.get_repr(products_)['cls_repr']
        # pickle.dump({'r_emb':reactants_emb,'p_emb':products_emb}, open('unimol_pred_emb.pkl','wb'))
        obj = pickle.load(open('unimol_pred_emb.pkl','rb'))
        reactants_emb = obj['r_emb'] ; products_emb = obj['p_emb']
        for i,r_emb in tqdm.tqdm(enumerate(reactants_emb), desc='predicting', total=len(reactants_emb)):
            s = i * BEAM_SIZE
            e = (i+1) * BEAM_SIZE
            product_emb = products_emb[s:e] if e < len(products_emb) else products_emb[s:]
            dist = self._calc_batch_dist(torch.tensor(r_emb), torch.tensor(product_emb))
            sorted_indices = np.argsort( -dist )
            sorted_product = np.array(products_list[i])[sorted_indices].tolist()
            sorted_products.append(sorted_product)
        return sorted_products
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.ckpt')
    parser.add_argument('--vocabulary_path', type=str, default='vocabulary.txt')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--sampled_test_path', type=str, default='sampled_test.pkl')
    args = parser.parse_args()
    set_seed()

    obj = pickle.load(open(args.sampled_test_path,'rb'))
    confidence_predictor = ConfidencePredictor(args.model_path, args.vocabulary_path, args.max_seq_len)
    unimol_predictor = UnimolPredictor()
    random_products = [np.random.permutation(products).tolist() for products in obj['sampled_products']]
    unimol_sorted_products = unimol_predictor(obj['reactants'], obj['sampled_products'].tolist())
    cofidence_sorted_products = confidence_predictor(obj['reactants'], obj['sampled_products'].tolist())
    
    # all_rankings = []
    # sorted_products = []
    # for reactant, sampled_product,target in tqdm.tqdm(zip(obj['reactants'], obj['sampled_products'], obj['target_products']), total=len(obj['reactants'])):
    #     try:
    #         r_emb, _ = model.transform([reactant])
    #         p_embs, _ = model.transform(sampled_product)
    #     except KeyboardInterrupt:
    #         exit()
    #     except:
    #         sorted_products.append(sampled_product.tolist())
    #         continue
    #     dist = cdist(r_emb, p_embs, metric='euclidean')[0]
    #     sorted_indices = np.argsort(dist)
    #     sorted_product = sampled_product[sorted_indices]
    #     sorted_products.append(sorted_product.tolist())

    # tmp = pickle.load(open('tmp.pkl','rb'))
    # sorted_products = tmp['sorted_products']; target_products = tmp['target_products']
    
    # compute_exact_match_accuracy(obj['sampled_products'], obj['target_products'], top_Ks=np.array([1,3,5,10,50]))
    # compute_exact_match_accuracy(sorted_products, obj['target_products'], top_Ks=np.array([1,3,5,10,50]))
    acc,_ = compute_substructure_accuracy(obj['sampled_products'], obj['target_products'], top_Ks=np.array([1,3,5,10,50]))
    print(f"original: {acc}")
    acc,_ = compute_substructure_accuracy(random_products, obj['target_products'], top_Ks=np.array([1,3,5,10,50]))
    print(f"random-ranked: {acc}")
    acc,_ = compute_substructure_accuracy(cofidence_sorted_products, obj['target_products'], top_Ks=np.array([1,3,5,10,50]))
    print(f"confidence re-ranked: {acc}")
    acc,_ = compute_substructure_accuracy(unimol_sorted_products, obj['target_products'], top_Ks=np.array([1,3,5,10,50]))
    print(f"unimol re-ranked: {acc}")