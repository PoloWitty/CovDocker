# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import math
import torch
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import warnings


@register_loss("finetune_cross_entropy")
class FinetuneCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = F.log_softmax(net_output.float(), dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = sample["target"]["finetune_target"].view(-1)
        loss = F.nll_loss(
            lprobs,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            probs = torch.cat([log.get("prob") for log in logging_outputs], dim=0)
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            if probs.size(-1) == 2:
                # binary classification task, add auc score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "probs": probs[:, 1].cpu(),
                        "targets": targets.cpu(),
                        "smi": smi_list,
                    }
                )
                auc = roc_auc_score(df["targets"], df["probs"])
                df = df.groupby("smi").mean()
                agg_auc = roc_auc_score(df["targets"], df["probs"])
                metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("multi_task_BCE")
class MultiTaskBCELoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            masked_tokens=None,
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        logit_output = net_output[0]
        is_valid = sample["target"]["finetune_target"] > -0.5
        loss = self.compute_loss(
            model, logit_output, sample, reduce=reduce, is_valid=is_valid
        )
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = torch.sigmoid(logit_output.float()).view(-1, logit_output.size(-1))
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "num_task": self.args.num_classes,
                "sample_size": sample_size,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, is_valid=None):
        pred = net_output[is_valid].float()
        targets = sample["target"]["finetune_target"][is_valid].float()
        loss = F.binary_cross_entropy_with_logits(
            pred,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            agg_auc_list = []
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("prob") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            # [test_size, num_classes] = [test_size * conf_size, num_classes].mean(axis=1)
            for i in range(y_true.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                    # ignore nan values
                    is_labeled = y_true[:, i] > -0.5
                    agg_auc_list.append(
                        roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                    )
            if len(agg_auc_list) < y_true.shape[1]:
                warnings.warn("Some target is missing!")
            if len(agg_auc_list) == 0:
                raise RuntimeError(
                    "No positively labeled data available. Cannot compute Average Precision."
                )
            agg_auc = sum(agg_auc_list) / len(agg_auc_list)
            metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("finetune_cross_entropy_pocket")
class FinetuneCrossEntropyPocketLoss(FinetuneCrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            preds = (
                torch.cat(
                    [log.get("prob").argmax(dim=-1) for log in logging_outputs], dim=0
                )
                .cpu()
                .numpy()
            )
            targets = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .cpu()
                .numpy()
            )
            metrics.log_scalar(f"{split}_pre", precision_score(targets, preds), round=3)
            metrics.log_scalar(f"{split}_rec", recall_score(targets, preds), round=3)
            metrics.log_scalar(
                f"{split}_f1", f1_score(targets, preds), sample_size, round=3
            )


class MultiTaskLoss(torch.nn.Module):
  '''https://arxiv.org/abs/1705.07115'''
  def __init__(self, is_regression, reduction='none'):
    '''
    is_regression should be a tensor of shape (n_tasks,) with 1 for regression tasks and 0 for classification tasks
    '''
    super(MultiTaskLoss, self).__init__()
    self.is_regression = is_regression
    self.n_tasks = len(is_regression)
    self.params = torch.nn.Parameter(torch.ones(self.n_tasks))
    torch.nn.init.trunc_normal_(self.params.data, mean=1.0, std=0.5, a=0, b=2.0) # modified (use trunc_normal init to avoid symmetry in the gradients)
    
  def forward(self, losses, reduction):
    '''weight loss by uncertainty'''
    dtype = losses.dtype
    device = losses.device
    stds = (self.params**2).to(device).to(dtype)
    self.is_regression = self.is_regression.to(device).to(dtype)
    coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
    
    # refer https://arxiv.org/pdf/1805.06334
    # use ln(1+sigma**2) instead of log(sigma**2) to avoid the loss of becoming negative during training
    multi_task_losses = coeffs*losses + torch.log(stds+1) 
    
    if reduction == 'sum':
      multi_task_losses = multi_task_losses.sum()
    if reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()

    return multi_task_losses


@register_loss("token_clf_cross_entropy_pocket")
class TokenClassificationCrossEntropyPocketLoss(FinetuneCrossEntropyLoss):
    """
        write for CovDocker bonded aa prediction
        author: Yangzhe Peng
        date: 2024/01/15
    """
    def __init__(self, task):
        super().__init__(task)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        pred_pocket_center, reactive_logit = net_output
        padding_mask = sample['net_input']['src_tokens'].eq(model.padding_idx)
        reactive_target = sample["target"]["finetune_target"]
        pocket_target = sample['target']['pocket_target'] # (b, n), 1 for pocket token, 0 for non-pocket token
        reduction = "mean" if reduce else "sum"
        
        # compute pocket center loss
        pocket_coord_target = sample['net_input']['src_coord'] * pocket_target.unsqueeze(-1) # (b,n,3)
        pocket_center_target = pocket_coord_target.sum(dim=1) / pocket_target.sum(dim=1) # (b,3)
        pocket_center_loss = F.huber_loss(pred_pocket_center,pocket_center_target,delta=model.args.pocket_coord_huber_delta, reduction=reduction)
    
        # compute reactive idx prediction loss
        reactive_logit_ = reactive_logit.masked_fill(~pocket_target.to(torch.bool),float('-inf')) # set non-pocket token to -inf
        assert torch.isneginf(reactive_logit_.index_select(index=reactive_target,dim=-1)).any() == False
        reactive_loss = F.cross_entropy(reactive_logit_, reactive_target, reduction=reduction) # (b,n) - (b,)
        
        # multi-task loss
        pocket_center_loss_ = model.args.pocket_coord_loss_weight * pocket_center_loss
        reactive_loss_ = model.args.reactive_loss_weight * reactive_loss
        # pocket_loss_ = model.args.pocket_token_clf_loss_weight * pocket_loss * (padding_mask.numel() / (~padding_mask).sum())
        loss = pocket_center_loss_ + reactive_loss_
        # loss = self.multi_task_loss(torch.stack([pocket_center_loss, reactive_loss]), reduction=reduction)
        
        # compute acc for reactive aa prediction
        reactive_pred = reactive_logit_.argmax(dim=-1)  # (b,)
        reactive_right_num = (reactive_pred == reactive_target).sum()
        
        if not self.training:
            # get the pocket tokens by 10A distance from pocket center (only for infer)
            coords = sample['net_input']['src_coord'] # (b,n,3)
            is_protein_mask = coords[:,:,0].ne(0).squeeze(-1)
            pred_pocket_center_ = pred_pocket_center.unsqueeze(1) # (b,1,3)
            dis = F.pairwise_distance(pred_pocket_center_, coords, p=2) # (b,n)
            infer_pocket_mask = dis < 10
            
            # calc infer acc for pocket token clf
            infer_pocket_clf_right_num  = ( (infer_pocket_mask == pocket_target) * (~padding_mask) ).sum()
            token_num = (~padding_mask).sum()
            infer_pocket_clf_acc = infer_pocket_clf_right_num / token_num
            
            # calc infer reactive acc
            infer_reactive_logit = reactive_logit.masked_fill(~infer_pocket_mask, 0)
            infer_reactive_idx = infer_reactive_logit.argmax(dim=-1)
            infer_reactive_right_num = (infer_reactive_idx == reactive_target).sum()
        
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            # tar_tokens = sample['net_input']['src_tokens'].gather(1,reactive_target.unsqueeze(1)).squeeze().cpu().numpy().tolist()
            # pred_tokens = sample['net_input']['src_tokens'].gather(1,reactive_pred.unsqueeze(1)).squeeze().cpu().numpy().tolist()
            infer_pocket_mask = infer_pocket_mask.data[is_protein_mask]
            res_ids = sample['res_ids']
            assert infer_pocket_mask.shape[0] == len(res_ids)
            logging_output = {
                "loss": loss.data.detach().cpu(),
                "pocket_center_loss": pocket_center_loss.detach().cpu(),
                "reactive_loss": reactive_loss.data.detach().cpu(),
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
                "reactive_right_num": reactive_right_num.data.detach().cpu(),
                "pocket_center_pred": pred_pocket_center.data.detach().cpu(),
                "pocket_center_target": pocket_center_target.data.detach().cpu(),
                "infer_pocket_clf_acc": infer_pocket_clf_acc.data.detach().cpu(),
                "infer_reactive_right_num": infer_reactive_right_num.data.detach().cpu(),
                "infer_pocket_mask": infer_pocket_mask.data.detach().cpu(),
                "protein_mask_target": sample['target']['pocket_target'][is_protein_mask].data.detach().cpu(),
                "pdb_id": sample["pdbid"],
                "res_ids": res_ids,
                # "pTokenClf_LW": self.multi_task_loss.params[0].data.detach().cpu(),
                # "pCenter_LW": self.multi_task_loss.params[0].data.detach().cpu(),
                # "reactive_LW": self.multi_task_loss.params[1].data.detach().cpu(),
            }
        else:
            logging_output = {
                "loss": loss.data.detach().cpu(),
                "reactive_loss": reactive_loss.data.detach().cpu(),
                "pocket_center_loss": pocket_center_loss.detach().cpu(),
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
                "reactive_right_num": reactive_right_num.data.detach().cpu(),
                "pocket_center_pred": pred_pocket_center.data.detach().cpu(),
                "pocket_center_target": pocket_center_target.data.detach().cpu(),
            }
        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        
        # pocket_loss_sum = sum(log.get("pocket_loss", 0) for log in logging_outputs)
        # metrics.log_scalar(
        #     "pocket_loss", pocket_loss_sum / sample_size / math.log(2), sample_size, round=3
        # )

        pocket_center_loss_sum = sum(log.get("pocket_center_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "pocket_center_loss", pocket_center_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        
        reactive_loss_sum = sum(log.get("reactive_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "reactive_loss", reactive_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        
        acc_sum = sum(log.get("reactive_right_num") for log in logging_outputs).cpu().numpy()
        size = sample_size.cpu().numpy() if type(sample_size)!=int else sample_size
        metrics.log_scalar(
            f"reactive_acc", acc_sum / size, size, round=3
        )
        
        # pocket_clf_acc_sum = sum(log.get("pocket_clf_acc") for log in logging_outputs).cpu().numpy()
        # metrics.log_scalar(
        #     f'pocket_clf_acc', pocket_clf_acc_sum / size, size, round=3
        # )
        
        if split == 'valid' or split == 'test':
            acc_sum = sum(log.get("infer_reactive_right_num") for log in logging_outputs).cpu().numpy()
            size = sample_size.cpu().numpy() if type(sample_size)!=int else sample_size
            metrics.log_scalar(
                f"infer_reactive_acc", acc_sum / size, size, round=3
            )
            
            pocket_clf_acc_sum = sum(log.get("infer_pocket_clf_acc") for log in logging_outputs).cpu().numpy()
            metrics.log_scalar(
                f'infer_pocket_clf_acc', pocket_clf_acc_sum / size, size, round=3
            )
            
            # pTokenClf_LW = sum(log.get("pTokenClf_LW") for log in logging_outputs).cpu().numpy()
            # metrics.log_scalar(
            #     f'pTokenClf_LW', pTokenClf_LW / len(logging_outputs), 1, round=3
            # )
            
            # pCenter_LW = sum(log.get("pCenter_LW") for log in logging_outputs).cpu().numpy()
            # metrics.log_scalar(
            #     f'pCenter_LW', pCenter_LW / len(logging_outputs), 1, round=3
            # )

            # reactive_LW = sum(log.get("reactive_LW") for log in logging_outputs).cpu().numpy()
            # metrics.log_scalar(
            #     f'reactive_LW', reactive_LW / len(logging_outputs), 1, round=3
            # )
        
        # calc pocket DCC
        pocket_center_preds = []; pocket_centers = []
        for log in logging_outputs:
            pocket_center_pred = log.get("pocket_center_pred")
            pocket_center = log.get("pocket_center_target")
            pocket_center_preds.append(pocket_center_pred)
            pocket_centers.append(pocket_center)
        pocket_center_pred = torch.cat(pocket_center_preds,dim=0)
        pocket_center = torch.cat(pocket_centers,dim=0)
        pocket_pairwise_dist = F.pairwise_distance(pocket_center_pred, pocket_center, p=2)
        DCC = (pocket_pairwise_dist < 4).sum().item() / len(pocket_pairwise_dist)
        metrics.log_scalar(
            f'pocket_DCC', DCC, 1, round=3
        )
        
        # if "valid" in split or "test" in split:
        #     pred_tokens = []
        #     tar_tokens = []
        #     for log in logging_outputs:
        #         if type(log.get("pred_tokens")) == int:
        #             pred_tokens.append(log.get("pred_tokens"))
        #             tar_tokens.append(log.get("tar_tokens"))
        #         else:
        #             pred_tokens += log.get("pred_tokens")
        #             tar_tokens += log.get("tar_tokens")
        #     with open('valid_res_last.csv','w') as f:
        #         f.write('pred,target\n')
        #         for i in range(len(pred_tokens)):
        #             f.write(str(pred_tokens[i])+','+str(tar_tokens[i])+'\n')
            
            # preds = (
            #     torch.cat(
            #         [log.get("prob").argmax(dim=-1) for log in logging_outputs], dim=0
            #     )
            #     .cpu()
            #     .numpy()
            # )
            # targets = (
            #     torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
            #     .cpu()
            #     .numpy()
            # )
            # metrics.log_scalar(f"{split}_pre", precision_score(targets, preds), round=3)
            # metrics.log_scalar(f"{split}_rec", recall_score(targets, preds), round=3)
            # metrics.log_scalar(
            #     f"{split}_f1", f1_score(targets, preds), sample_size, round=3
            # )

