# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
import torch


@register_loss("docking_pose_cov")
class DockingPossCovLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.padding_idx = task.dictionary.pad()
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_outputs = model(**sample["net_input"])
        cross_distance_predict, holo_distance_predict = net_outputs[0], net_outputs[1]
        sample_size = sample["target"]["holo_coord"].size(0)

        ### distance loss
        # change according to https://github.com/dptech-corp/Uni-Mol/commit/73946479da134ba025567ff329c3501ccd2e4591
        distance_mask = sample["target"]["distance_target"].ne(0)  # 0 for padding, BOS and EOS
        # 0 is impossible in the cross distance matrix, all the relevant cross distances are kept
        
        inter_bond = sample['inter_bond'][:,:2] # dataset will pad it to 8 columns
        orig2cropped = sample["orig2cropped_pocket"]
        # map orig index to cropped index
        inter_bond[:,0] = orig2cropped[torch.arange(sample_size), inter_bond[:,0]]
        inter_bond += 1 # add 1 for bos
        inter_bond_dist = cross_distance_predict[torch.arange(sample_size),inter_bond[:,1],inter_bond[:,0]] # (b,)
        inter_bond_dist = inter_bond_dist.unsqueeze(-1).unsqueeze(-1) # (b,1,1)
        
        if self.args.dist_threshold > 0:
            distance_mask &= (
                sample["target"]["distance_target"] < self.args.dist_threshold
            )

        if self.args.min_dist_norm_type==1:
            min_dist_norm_loss =  torch.nn.functional.relu((inter_bond_dist - cross_distance_predict)[distance_mask])
            min_dist_norm_loss = min_dist_norm_loss.mean()
        elif self.args.min_dist_norm_type==2:# will push predicted distance to 0 or 1 (logical error when design)
            norm_one_hot_target = torch.zeros_like(cross_distance_predict)
            norm_one_hot_target[torch.arange(sample_size),inter_bond[:,1],inter_bond[:,0]] = 1
            norm_one_hot_target_ = torch.argmax(norm_one_hot_target.view(sample_size,-1),dim=-1) # bond place will be 1, otherwise 0
            cross_distance_predict_ = cross_distance_predict.masked_fill(~distance_mask, float('inf'))
            cross_distance_predict_[torch.arange(sample_size),inter_bond[:,1],inter_bond[:,0]] = cross_distance_predict[torch.arange(sample_size),inter_bond[:,1],inter_bond[:,0]] # in case inter_bond position is masked
            min_dist_norm_loss = torch.nn.functional.cross_entropy((-cross_distance_predict_).view(sample_size,-1), norm_one_hot_target_,ignore_index=-100,reduction='mean')
        elif self.args.min_dist_norm_type==0:
            cross_distance_predict_ = cross_distance_predict.masked_fill(~distance_mask, float('inf'))
            pred_min_dist = cross_distance_predict_.view(sample_size, -1).min(dim=-1)[0]
            min_dist_norm_loss =  (inter_bond_dist.squeeze() - pred_min_dist).abs()
            min_dist_norm_loss = min_dist_norm_loss.mean()
        else:
            raise NotImplementedError

        distance_predict = cross_distance_predict[distance_mask]
        distance_target = sample["target"]["distance_target"][distance_mask]
        distance_loss = F.mse_loss(
            distance_predict.float(), distance_target.float(), reduction="mean"
        )
        if torch.isnan(distance_loss).sum():
            distance_loss[torch.isnan(distance_loss)] = 0
        ### holo distance loss
        token_mask = sample["net_input"]["mol_src_tokens"].ne(self.padding_idx) & \
                     sample["net_input"]["mol_src_tokens"].ne(self.eos_idx) & \
                     sample["net_input"]["mol_src_tokens"].ne(self.bos_idx)
        holo_distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
        holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
        holo_distance_target = sample["target"]["holo_distance_target"][
            holo_distance_mask
        ]
        holo_distance_loss = F.smooth_l1_loss(
            holo_distance_predict_train.float(),
            holo_distance_target.float(),
            reduction="mean",
            beta=1.0,
        )

        if self.args.min_dist_norm_weight > 0:
            loss = distance_loss + holo_distance_loss + self.args.min_dist_norm_weight * min_dist_norm_loss
        else:
            loss = distance_loss + holo_distance_loss
        logging_output = {
            "loss": loss.data.detach().cpu(),
            "cross_loss": distance_loss.data.detach().cpu(),
            "holo_loss": holo_distance_loss.data.detach().cpu(),
            "min_dist_norm_loss": min_dist_norm_loss.data.detach().cpu(),
            "bsz": sample_size,
            "sample_size": 1,
        }
        if not self.training:
            logging_output["smi_name"] = sample["smi_name"]
            logging_output["pocket_name"] = sample["pocket_name"]
            logging_output[
                "cross_distance_predict"
            ] = cross_distance_predict.data.detach().cpu()
            logging_output[
                "holo_distance_predict"
            ] = holo_distance_predict.data.detach().cpu()
            logging_output["atoms"] = (
                sample["net_input"]["mol_src_tokens"].data.detach().cpu()
            )
            logging_output["pocket_atoms"] = (
                sample["net_input"]["pocket_src_tokens"].data.detach().cpu()
            )
            logging_output["holo_center_coordinates"] = (
                sample["holo_center_coordinates"].data.detach().cpu()
            )
            logging_output["holo_coordinates"] = (
                sample["target"]["holo_coord"].data.detach().cpu()
            )
            logging_output["pocket_coordinates"] = (
                sample["net_input"]["pocket_src_coord"].data.detach().cpu()
            )
            logging_output["orig2cropped_pocket"] = (
                sample["orig2cropped_pocket"].data.detach().cpu()
            )
            logging_output["cross_distance_target"] = sample["target"]["distance_target"].data.detach().cpu()
            logging_output["holo_distance_target"] = sample["target"]["holo_distance_target"].data.detach().cpu()

        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=4)
        metrics.log_scalar(
            f"{split}_loss", loss_sum / sample_size, sample_size, round=4
        )
        cross_loss_sum = sum(log.get("cross_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "cross_loss", cross_loss_sum / sample_size, sample_size, round=4
        )
        holo_loss_sum = sum(log.get("holo_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "holo_loss", holo_loss_sum / sample_size, sample_size, round=4
        )
        min_dist_norm_loss_sum = sum(log.get("min_dist_norm_loss", 0) for log in logging_outputs)
        metrics.log_scalar(
            "min_dist_norm_loss", min_dist_norm_loss_sum / sample_size, sample_size, round=4
        )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
