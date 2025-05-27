
import pickle
import argparse
import torch
from tabulate import tabulate
import json
try:
    import wandb
except:
    pass

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-file", type=str, help="cov docking position prediction infer result")
    parser.add_argument("--run-id", type=str, default='default-run-id')
    parser.add_argument("--use-wandb",type=int, default=0, help='whether use wandb to store result')
    parser.add_argument("--infer-config-filename", type=str, default='' ,help="infer config filename to log")
    args = parser.parse_args()
    
    data = pickle.load(open(args.predict_file, 'rb'))
    
    right_sum = 0
    total_num = 0
    pocket_center_pred = []; pocket_center_gt = []
    for batch in data:
        right_sum += batch['infer_reactive_right_num'].cpu().item()
        total_num += batch['bsz']
        pocket_center_pred.append( batch['pocket_center_pred'] )
        pocket_center_gt.append( batch['pocket_center_target'] )
    
        # pred_tokens = batch['pred_tokens']
        # tar_tokens = batch['tar_tokens']
        
    pocket_center_pred = torch.cat(pocket_center_pred,dim=0)
    pocket_center_gt = torch.cat(pocket_center_gt,dim=0)
    pocket_pairwise_dist = torch.nn.functional.pairwise_distance(pocket_center_pred, pocket_center_gt, p=2)
    DCC_3 = (pocket_pairwise_dist < 3).sum().item() / len(pocket_pairwise_dist)
    DCC_4 = (pocket_pairwise_dist < 4).sum().item() / len(pocket_pairwise_dist)
    DCC_5 = (pocket_pairwise_dist < 5).sum().item() / len(pocket_pairwise_dist)
    
    table = {
        'id': args.run_id,
        'Infer reactive Accuracy:': right_sum/total_num,
        'DCC_3:': DCC_3,
        'DCC_4:': DCC_4,
        'DCC_5:': DCC_5
    }
    
    
    if args.use_wandb:
        wandb.init(
            project="res_reactive_site",
            config = json.load(open(args.infer_config_filename))
        )
        wandb.config.update(args)
        wandb.log(table)


    # use tabulate to show the result
    table_ = {k:[v] for k,v in table.items()}
    print(tabulate(table_, headers="keys"))