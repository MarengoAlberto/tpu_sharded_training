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
    def __init__(self, num_classes=2, alpha=0.25, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.num_classes  = int(num_classes)
        self.alpha        = float(alpha)
        self.gamma        = float(gamma)
        self.ignore_index = int(ignore_index)

    def forward(self, cls_preds, cls_targets):
        """
        cls_preds:   [N, A, C] logits
        cls_targets: [N, A]     in {ignore_index, 0..C-1}
        """
        N, A, C = cls_preds.shape
        tgt = cls_targets.long()                     # [N, A]
        valid = (tgt != self.ignore_index)           # [N, A]
        pos   = (tgt > 0) & valid                    # [N, A]

        # Cross-entropy per anchor (keep fixed shape via one-hot)
        # Convert targets to one-hot but zero out ignored
        one_hot = F.one_hot(tgt.clamp_min(0), num_classes=C).to(cls_preds.dtype)  # [N,A,C]
        one_hot = one_hot * valid.unsqueeze(-1)                                   # ignore -> 0

        log_probs = F.log_softmax(cls_preds, dim=-1)            # [N,A,C]
        ce = -(one_hot * log_probs).sum(dim=-1)                 # [N,A]

        # p_t = prob of the target class per anchor (masked)
        probs = log_probs.exp()                                 # [N,A,C]
        p_t   = (probs * one_hot).sum(dim=-1) + (~valid).to(probs.dtype)  # [N,A]; ignored -> 1 so weight→0

        # alpha per anchor (pos vs bg)
        alpha_t = torch.where(pos,
                              torch.full_like(p_t, self.alpha),
                              torch.full_like(p_t, 1.0 - self.alpha))

        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * ce      # [N,A]
        num_pos = pos.sum().clamp(min=1).to(focal.dtype)
        loss = (focal * valid).sum() / num_pos
        return torch.nan_to_num(loss)


class IoULoss(nn.Module):
    def __init__(self, iou_type: str = "ciou", encoded: bool = False,
                 anchors: torch.Tensor = None, variances=(0.1, 0.2), eps=1e-7):
        super().__init__()
        assert iou_type in {"giou", "ciou"}
        self.iou_type = iou_type
        self.encoded  = encoded
        self.anchors  = anchors   # [A,4] (cx,cy,w,h) if encoded=True
        self.v0, self.v1 = float(variances[0]), float(variances[1])
        self.eps = float(eps)

    @staticmethod
    def _xywh_to_xyxy(xywh):
        cx, cy, w, h = xywh.unbind(-1)
        x1 = cx - 0.5 * w; y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w; y2 = cy + 0.5 * h
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def _decode(self, deltas, anchors):
        dx, dy, dw, dh = deltas.unbind(-1)
        acx, acy, aw, ah = anchors.unbind(-1)
        px = acx + dx * aw * self.v0
        py = acy + dy * ah * self.v0
        pw = aw * torch.exp(dw * self.v1)
        ph = ah * torch.exp(dh * self.v1)
        return torch.stack((px, py, pw, ph), dim=-1)

    def _giou_ciou(self, p_xyxy, t_xyxy):
        # all ops are elementwise on [N, A, 4]
        x1p, y1p, x2p, y2p = p_xyxy.unbind(-1)
        x1t, y1t, x2t, y2t = t_xyxy.unbind(-1)

        wp = (x2p - x1p).clamp(min=self.eps)
        hp = (y2p - y1p).clamp(min=self.eps)
        wt = (x2t - x1t).clamp(min=self.eps)
        ht = (y2t - y1t).clamp(min=self.eps)
        area_p = wp * hp
        area_t = wt * ht

        x1i = torch.max(x1p, x1t); y1i = torch.max(y1p, y1t)
        x2i = torch.min(x2p, x2t); y2i = torch.min(y2p, y2t)
        wi = (x2i - x1i).clamp(min=0)
        hi = (y2i - y1i).clamp(min=0)
        inter = wi * hi
        union = area_p + area_t - inter + self.eps
        iou = inter / union

        if self.iou_type == "giou":
            x1c = torch.min(x1p, x1t); y1c = torch.min(y1p, y1t)
            x2c = torch.max(x2p, x2t); y2c = torch.max(y2p, y2t)
            wc = (x2c - x1c).clamp(min=self.eps)
            hc = (y2c - y1c).clamp(min=self.eps)
            area_c = wc * hc
            giou = iou - (area_c - union) / area_c
            return 1.0 - giou

        # CIoU terms
        cxp = 0.5 * (x1p + x2p); cyp = 0.5 * (y1p + y2p)
        cxt = 0.5 * (x1t + x2t); cyt = 0.5 * (y1t + y2t)
        rho2 = (cxp - cxt) ** 2 + (cyp - cyt) ** 2

        x1c = torch.min(x1p, x1t); y1c = torch.min(y1p, y1t)
        x2c = torch.max(x2p, x2t); y2c = torch.max(y2p, y2t)
        c2 = ((x2c - x1c) ** 2 + (y2c - y1c) ** 2).clamp(min=self.eps)

        v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(wt / ht) - torch.atan(wp / hp), 2)
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + self.eps)
        ciou = iou - (rho2 / c2) - alpha * v
        return 1.0 - ciou

    def forward(self, loc_preds, loc_targets, cls_targets):
        """
        loc_*: [N, A, 4] (xyxy if encoded=False; deltas if encoded=True)
        cls_targets: [N, A] (0=bg, >0=pos, -1=ignore)
        """
        N, A, _ = loc_preds.shape
        tgt = cls_targets.long()
        pos = (tgt > 0)                                      # [N, A]
        num_pos = pos.sum().clamp(min=1).to(loc_preds.dtype)

        if self.encoded:
            assert self.anchors is not None and self.anchors.shape[0] == A
            anchors = self.anchors.to(loc_preds.device).unsqueeze(0).expand(N, A, 4)
            p_xywh = self._decode(loc_preds, anchors)
            t_xywh = self._decode(loc_targets, anchors)
            p_xyxy = self._xywh_to_xyxy(p_xywh)
            t_xyxy = self._xywh_to_xyxy(t_xywh)
        else:
            p_xyxy = loc_preds
            t_xyxy = loc_targets

        # Compute per-anchor loss with fixed shape, then mask positives
        per_anchor = self._giou_ciou(p_xyxy, t_xyxy)         # [N, A]
        loss = (per_anchor * pos.to(per_anchor.dtype)).sum() / num_pos
        return torch.nan_to_num(loss)
