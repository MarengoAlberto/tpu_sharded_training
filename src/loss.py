import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class SmoothL1Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loc_loss_fn = nn.SmoothL1Loss(reduction="none")

    def forward(self, loc_preds, loc_targets, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets).
        '''

        ################################################################
        # loc_loss
        ################################################################

        cls_targets = cls_targets.long()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.sum() # Scalar
        mask = pos.unsqueeze(dim=2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [num_pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [num_pos,4]

        loc_loss = self.loc_loss_fn(masked_loc_preds, masked_loc_targets)
        loc_loss = torch.nan_to_num(loc_loss.sum() / num_pos.float())

        return loc_loss


class OHEMLoss(nn.Module):
    def __init__(self, num_classes=2, neg2pos_ratio=3, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.negpos_ratio = neg2pos_ratio
        self.cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = OHEMLoss(cls_preds, cls_targets).
        '''

        ################################################################
        # cls_loss
        ################################################################

        cls_targets = cls_targets.long()
        pos = cls_targets > 0  # [N, #anchors]
        num_pos_per_image_batch = pos.sum(dim=1, keepdim=True) # [N, 1]
        total_pos = num_pos_per_image_batch.sum().float().clamp(min=1.0) # Scalar

        cls_preds_reshaped = cls_preds.permute(dims=(0,2,1)) #[N, #classes, #anchors]
        cls_loss = self.cls_loss_fn(cls_preds_reshaped, cls_targets)  # [N, #anchors]

        pos_cls_loss = cls_loss[pos] # [#total_positives, ]
        cls_loss[pos] = 0

        _, loss_idx = cls_loss.sort(dim=1, descending=True) # [N, #anchors]

        _, idx_rank = loss_idx.sort(dim=1) # [N, #anchors]

        num_neg_per_image_batch = torch.clamp(self.negpos_ratio * num_pos_per_image_batch, min=1, max=pos.shape[1] - 1) # [N, 1]

        neg = idx_rank < num_neg_per_image_batch.expand_as(idx_rank) # [N, #anchors]
        neg_cls_loss = cls_loss[neg]  # [#total_negatives, ]

        cls_loss = (pos_cls_loss.sum() + neg_cls_loss.sum()) / total_pos


        return cls_loss

    

# -----------------------------
# Classification: Focal Loss
# -----------------------------
class FocalLoss(nn.Module):
    """
    Focal loss for dense detection.
    Accepts either softmax logits (C>=2) or a single sigmoid logit (C=1).

    Args:
        num_classes: number of classes. For binary softmax use 2.
        alpha: weight for the positive class (e.g., 0.25).
        gamma: focusing parameter (e.g., 2.0).
        ignore_index: targets equal to this are ignored.
    """
    def __init__(self, num_classes=2, alpha=0.25, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ignore_index = ignore_index

    def forward(self, cls_preds, cls_targets):
        """
        Args:
            cls_preds: [N, A, C] logits.
            cls_targets: [N, A] with values in {ignore_index, 0..C-1}.
                         For binary sigmoid, targets should be {0,1} or {ignore,0,1}.
        Returns:
            cls_loss: scalar
        """
        N, A, C = cls_preds.shape[0], cls_preds.shape[1], cls_preds.shape[2]
        targets = cls_targets.long().view(-1)                 # [N*A]
        logits  = cls_preds.view(-1, C)                       # [N*A, C]

        valid_mask = (targets != self.ignore_index)           # [N*A]
        if valid_mask.sum() == 0:
            return logits.new_tensor(0.0)

        # normalize by number of positives (like common one-stage practice)
        pos_mask = (targets == 1) if C >= 2 else (targets == 1)
        num_pos  = torch.clamp(pos_mask[valid_mask].sum().float(), min=1.0)

        if C >= 2:
            # ----- Softmax focal -----
            # CE per anchor
            ce = F.cross_entropy(logits[valid_mask], targets[valid_mask], reduction="none")  # [Nv]
            with torch.no_grad():
                probs = F.softmax(logits[valid_mask], dim=-1)                                 # [Nv, C]
                p_t = probs[torch.arange(probs.size(0), device=probs.device), targets[valid_mask]]  # [Nv]
                # alpha for positive (class 1); 1-alpha for background (class 0)
                alpha_t = torch.where(targets[valid_mask] == 1,
                                      torch.full_like(p_t, self.alpha),
                                      torch.full_like(p_t, 1.0 - self.alpha))
            loss = (alpha_t * (1.0 - p_t).pow(self.gamma) * ce).sum() / num_pos
            return torch.nan_to_num(loss)

        else:
            # ----- Binary sigmoid focal (C == 1) -----
            # logits: [N*A, 1] -> [N*A]
            logits_bin = logits.squeeze(-1)
            targets_bin = targets.float()
            ce = F.binary_cross_entropy_with_logits(logits_bin[valid_mask], targets_bin[valid_mask], reduction="none")
            with torch.no_grad():
                p = torch.sigmoid(logits_bin[valid_mask])
                p_t = p * targets_bin[valid_mask] + (1 - p) * (1 - targets_bin[valid_mask])
                alpha_t = self.alpha * targets_bin[valid_mask] + (1 - self.alpha) * (1 - targets_bin[valid_mask])
            loss = (alpha_t * (1.0 - p_t).pow(self.gamma) * ce).sum() / num_pos
            return torch.nan_to_num(loss)


# -----------------------------
# Regression: IoU (CIoU/GIoU)
# -----------------------------
class IoULoss(nn.Module):
    """
    IoU-based regression loss: CIoU (default) or GIoU.

    Usage 1 (preferred): pass decoded boxes (xyxy) to forward().
        loc_preds/loc_targets shape: [N, A, 4] in xyxy.

    Usage 2: if you only have encoded deltas relative to anchors,
             provide anchors (cx,cy,w,h) to the constructor and set encoded=True.
             The class will decode deltas using Retina/SSD-style parameterization:
                x = ax + dx * aw * v0
                y = ay + dy * ah * v0
                w = aw * exp(dw * v1)
                h = ah * exp(dh * v1)
    """
    def __init__(self, iou_type: str = "ciou", encoded: bool = False,
                 anchors: torch.Tensor = None, variances=(0.1, 0.2), eps=1e-7):
        super().__init__()
        assert iou_type in {"giou", "ciou"}, "iou_type must be 'giou' or 'ciou'"
        self.iou_type = iou_type
        self.encoded = encoded
        self.anchors = anchors  # Tensor [A,4] in (cx,cy,w,h) if encoded=True
        self.v0, self.v1 = float(variances[0]), float(variances[1])
        self.eps = float(eps)

    @staticmethod
    def _xywh_to_xyxy(box):
        cx, cy, w, h = box.unbind(-1)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _decode(self, deltas, anchors):
        # deltas, anchors: [..., 4] with anchors in cx,cy,w,h
        dx, dy, dw, dh = deltas.unbind(-1)
        acx, acy, aw, ah = anchors.unbind(-1)
        px = acx + dx * aw * self.v0
        py = acy + dy * ah * self.v0
        pw = aw * torch.exp(dw * self.v1)
        ph = ah * torch.exp(dh * self.v1)
        return torch.stack((px, py, pw, ph), dim=-1)

    def _giou_ciou_loss(self, pred_xyxy, tgt_xyxy):
        # pred_xyxy/tgt_xyxy: [M,4] (only positives)
        x1p, y1p, x2p, y2p = pred_xyxy.unbind(-1)
        x1t, y1t, x2t, y2t = tgt_xyxy.unbind(-1)

        # areas
        wp = (x2p - x1p).clamp(min=self.eps)
        hp = (y2p - y1p).clamp(min=self.eps)
        wt = (x2t - x1t).clamp(min=self.eps)
        ht = (y2t - y1t).clamp(min=self.eps)
        area_p = wp * hp
        area_t = wt * ht

        # intersection
        x1i = torch.max(x1p, x1t)
        y1i = torch.max(y1p, y1t)
        x2i = torch.min(x2p, x2t)
        y2i = torch.min(y2p, y2t)
        wi = (x2i - x1i).clamp(min=0)
        hi = (y2i - y1i).clamp(min=0)
        inter = wi * hi

        union = area_p + area_t - inter + self.eps
        iou = inter / union

        if self.iou_type == "giou":
            # enclosing box
            x1c = torch.min(x1p, x1t)
            y1c = torch.min(y1p, y1t)
            x2c = torch.max(x2p, x2t)
            y2c = torch.max(y2p, y2t)
            wc = (x2c - x1c).clamp(min=self.eps)
            hc = (y2c - y1c).clamp(min=self.eps)
            area_c = wc * hc
            giou = iou - (area_c - union) / area_c
            loss = 1.0 - giou
            return loss

        # CIoU (includes distance + aspect terms)
        # center distance
        cxp = (x1p + x2p) * 0.5
        cyp = (y1p + y2p) * 0.5
        cxt = (x1t + x2t) * 0.5
        cyt = (y1t + y2t) * 0.5
        rho2 = (cxp - cxt) ** 2 + (cyp - cyt) ** 2

        # diagonal length of smallest enclosing box
        x1c = torch.min(x1p, x1t)
        y1c = torch.min(y1p, y1t)
        x2c = torch.max(x2p, x2t)
        y2c = torch.max(y2p, y2t)
        c2 = ((x2c - x1c) ** 2 + (y2c - y1c) ** 2).clamp(min=self.eps)

        # aspect ratio consistency
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(wt / ht) - torch.atan(wp / hp), 2)
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + self.eps)

        ciou = iou - (rho2 / c2) - alpha * v
        loss = 1.0 - ciou
        return loss

    def forward(self, loc_preds, loc_targets, cls_targets):
        """
        Args:
            loc_preds:  [N, A, 4] (xyxy if encoded=False; deltas if encoded=True)
            loc_targets:[N, A, 4] (xyxy if encoded=False; deltas if encoded=True)
            cls_targets:[N, A] (0=bg, >0=pos, -1=ignore)
        Returns:
            loc_loss: scalar
        """
        N, A, _ = loc_preds.shape
        cls_targets = cls_targets.long()
        pos = (cls_targets > 0)                                   # [N,A]
        num_pos = torch.clamp(pos.sum().float(), min=1.0)

        if pos.sum() == 0:
            return loc_preds.new_tensor(0.0)

        if self.encoded:
            assert self.anchors is not None and self.anchors.shape[0] == A, \
                "Provide anchors [A,4] in (cx,cy,w,h) to decode deltas"
            anchors = self.anchors.to(loc_preds.device).unsqueeze(0).expand(N, A, 4)  # [N,A,4]
            pred_xywh = self._decode(loc_preds, anchors)   # [N,A,4] (cxcywh)
            tgt_xywh  = self._decode(loc_targets, anchors)
            pred = self._xywh_to_xyxy(pred_xywh)[pos].view(-1, 4)
            tgt  = self._xywh_to_xyxy(tgt_xywh)[pos].view(-1, 4)
        else:
            pred = loc_preds[pos].view(-1, 4)               # xyxy
            tgt  = loc_targets[pos].view(-1, 4)             # xyxy

        loss_vec = self._giou_ciou_loss(pred, tgt)          # [num_pos]
        loss = torch.nan_to_num(loss_vec.sum() / num_pos)
        return loss
