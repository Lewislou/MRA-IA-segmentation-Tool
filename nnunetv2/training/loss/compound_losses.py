import torch
import torch.nn.functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables '\
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()

            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)\
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long())\
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:

            mask = (1 - target[:, -1:]).bool()

            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables '\
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()

            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)\
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target)\
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = None, reduction: str = 'mean',
                 eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()

        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
        else:
            valid = None

        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)

        gather_index = target.unsqueeze(1)
        pt = torch.gather(p, dim=1, index=gather_index).squeeze(1).clamp_min(self.eps)
        logpt = torch.gather(logp, dim=1, index=gather_index).squeeze(1)

        loss = - (self.alpha * (1 - pt) ** self.gamma * logpt)
        if valid is not None:
            loss = loss[valid]

        if self.reduction == 'mean':
            return loss.mean() if loss.numel() > 0 else loss.sum()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class DC_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_focal=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        if ignore_label is not None and 'ignore_index' not in focal_kwargs:
            focal_kwargs = dict(focal_kwargs)
            focal_kwargs['ignore_index'] = ignore_label

        self.focal = SoftmaxFocalLoss(**focal_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label not implemented for one hot encoded targets (DC_and_Focal_loss)'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        focal_loss = self.focal(net_output, target) if self.weight_focal != 0 and (self.ignore_label is None or num_fg > 0) else 0
        return self.weight_focal * focal_loss + self.weight_dice * dc_loss

class SMDNetMultiTaskLoss(nn.Module):
    def __init__(self, aneurysm_loss, vessel_loss, lambda_v: float = 1.0, lambda_a: float = 1.0):
        super().__init__()
        self.aneurysm_loss = aneurysm_loss
        self.vessel_loss = vessel_loss
        self.lambda_v = float(lambda_v)
        self.lambda_a = float(lambda_a)

    def forward(self, fine_outputs, coarse_logits, target):
        l_a = self.aneurysm_loss(fine_outputs, target)

        tgt0 = target[0] if isinstance(target, (list, tuple)) else target

        vessel_tgt = (tgt0 > 0).long()
        if vessel_tgt.shape[2:] != coarse_logits.shape[2:]:
            vessel_tgt = F.interpolate(
                vessel_tgt.float(), size=coarse_logits.shape[2:], mode='nearest'
            ).long()
        l_v = self.vessel_loss(coarse_logits, vessel_tgt)
        return self.lambda_v * l_v + self.lambda_a * l_a
