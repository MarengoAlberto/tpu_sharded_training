import math
import torch
from torchvision.ops import nms


class DataEncoder:
    def __init__(self, input_size=(300, 300), classes=("__background__", "person")):
        self.input_size = input_size
        self.anchor_boxes, self.aspect_ratios, self.scales = get_all_anchor_boxes(input_size=self.input_size)
        self.classes = classes

    def encode(self, boxes, classes):
        if boxes.shape[0] == 0 and classes.shape[0] == 0:
            return  (torch.zeros((self.anchor_boxes.shape[0], 4),
                                device=self.anchor_boxes.device,
                                dtype=self.anchor_boxes.dtype),
                     torch.squeeze(torch.zeros((self.anchor_boxes.shape[0], 1),
                                 device=self.anchor_boxes.device,
                                 dtype=self.anchor_boxes.dtype)))
        iou = compute_iou(src=boxes, dst=self.anchor_boxes)
        iou, ids = iou.max(dim=1)

        cls_targets = classes[ids]
        cls_targets[iou < 0.5] = -1
        cls_targets[iou < 0.3] =  0

        loc_targets = encode_boxes(boxes=boxes[ids], anchors=self.anchor_boxes)
        return loc_targets, cls_targets

    def decode(self,
               loc_pred,
               cls_pred,
               device,
               nms_threshold=0.5,
               score_threshold=0.5,
               max_dets=100,
               ):


        input_h, input_w = self.input_size[:2]

        min_size_clamp = torch.tensor([0.,0.,0.,0.], device=device)
        max_size_clamp = torch.tensor([input_w, input_h, input_w, input_h], device=device)


        self.anchor_boxes = self.anchor_boxes.to(device)

        # loc_pred shape: [#anchors, 4], # cls_pred shape: [#anchors, #num_classes]
        pred_boxes = decode_boxes(deltas=loc_pred, anchors=self.anchor_boxes) #shape: [#anchors, 4]

        pred_boxes = torch.clamp(pred_boxes, min=min_size_clamp, max=max_size_clamp)

        pred_confs = cls_pred.softmax(dim=1) #shape: [#anchors, #num_classes]

        # Perform Argmax
        max_conf_score, conf_argmax = pred_confs.max(dim=1, keepdim=True) #shape: [#anchors, 1]

        # Combined Tensor: shape [#anchors ,6].
        # 6: [xmin, ymin, xmax, ymax, conf_score, class_id]
        combined_tensor = torch.cat([pred_boxes, max_conf_score, conf_argmax], dim=1)

        # Store final boxes that needs to be retained.
        chosen_boxes = []

        for cls_idx, cls_name in enumerate(self.classes):

            if cls_name == "__background__":
                continue

            # Get current class ID from comnined_tensor
            class_ids = torch.where(combined_tensor[:,5].int() == cls_idx)[0]

            class_tensor = combined_tensor[class_ids] #shape: [#class_ids, 6]
            class_boxes = class_tensor[:,:4]
            class_conf = class_tensor[:,4]

            keep = nms(boxes=class_boxes, scores=class_conf, iou_threshold=nms_threshold)
            # keep = soft_nms(class_boxes, class_conf, sigma=nms_threshold, score_thresh=0.001, topk=300) #score_threshold

            filtered_ids = torch.where(class_conf[keep] > score_threshold)[0]

            # Final boxes and conf. scores to be retained for the current class
            # after NMS.
            # The number of final boxes is constrained by max_dets.
            final_box_data = class_tensor[keep][filtered_ids][:max_dets]

            chosen_boxes.append(final_box_data)


        return torch.cat(chosen_boxes)

    def get_num_anchors(self):
        return len(self.aspect_ratios) * len(self.scales)


def get_all_anchor_boxes(input_size, anchor_areas=None, aspect_ratios=None, scales=None):

    if anchor_areas is None:
        anchor_areas = [16**2, 32**2, 64**2, 128**2, 256**2, 512**2]
        # [
        #     8 * 8,  # For feature_map size: 38x38
        #     16 * 16.0,  # For feature_map size: 19x19
        #     32 * 32.0,  # For feature_map size: 10x10
        #     64 * 64.0,  # For feature_map size: 5x5
        #     128 * 128,  # For feature_map size: 3x3
        # ]  # p3 -> p7

    if aspect_ratios is None:
        aspect_ratios = [5.0, 3.0, 2.0, 1.0] #[4.0, 3.0, 2.0, 1.5]  #[1.19, 1.81, 2.45, 3.82]  #[0.5, 1.0, 2.0]

    if scales is None:
        scales = [
                    [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P2 [0.1, 0.25, 0.5, 1.0]
                    [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P3 (small/med) [0.1, 0.25, 0.5, 1.0]
                    [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P4 
                    [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P5 [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)]
                    [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P6  [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)]
                    [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)],  # P7  [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)]
                ]  
            # [0.5, 1, pow(2, 1 / 3.0), pow(2, 4.5 / 3.0)] #[1.0, 2**(1/3), 2**(2/3)]
            #[1, pow(2, 1 / 3.0), pow(2, 3.5 / 3.0)] #[1, pow(2, 1 / 3.0), pow(2, 2 / 3.0)]

    num_fms = len(anchor_areas)
    # fm_sizes = [math.ceil(input_size[0] / pow(2.0, i + 3)) for i in range(num_fms)]
    strides = [4, 8, 16, 32, 64, 128][:num_fms]
    fm_sizes = [math.ceil(input_size[0] / s) for s in strides]

    anchor_boxes = []

    for idx, fm_size in enumerate(fm_sizes):
        anchors = generate_anchors(anchor_areas[idx], aspect_ratios, scales[idx])
        anchor_grid = generate_anchor_grid(input_size, fm_size, anchors)
        anchor_boxes.append(anchor_grid)

    anchor_boxes = torch.concat(anchor_boxes, dim=0)

    return anchor_boxes, aspect_ratios, scales


def generate_anchors(anchor_area, aspect_ratios, scales):
    anchors = []

    for scale in scales:
        for ratio in aspect_ratios:

            h = scale * (math.sqrt(anchor_area / ratio))  # h*w * h/w = sqrt(h**2)
            w = ratio * h  # w/h * h

            # Assume the anchor box is centered at origin (0, 0)
            # Get xmin, ymin, xmax, ymax of anchor box w.r.t origin (0, 0)
            box_w_half = w / 2
            box_h_half = h / 2

            x1 = 0.0 - box_w_half
            y1 = 0.0 - box_h_half

            x2 = 0.0 + box_w_half
            y2 = 0.0 + box_h_half

            anchors.append([x1, y1, x2, y2])

    return torch.tensor(anchors, dtype=torch.float)


def generate_anchor_grid(input_size, fm_size, anchors):
    img_h, img_w = input_size

    grid_h = math.ceil(img_h / fm_size)
    grid_w = math.ceil(img_w / fm_size)

    grid_h_coords = torch.arange(0, fm_size, dtype=torch.float) * grid_h + grid_h / 2
    grid_w_coords = torch.arange(0, fm_size, dtype=torch.float) * grid_w + grid_w / 2

    # Create a numpy-like cartesian meshgrid.
    x, y = torch.meshgrid(grid_w_coords, grid_h_coords, indexing="xy")

    xyxy = torch.stack([x, y, x, y], dim=2)
    anchors = anchors.reshape(-1, 1, 1, 4)
    boxes = (xyxy + anchors).permute(1, 2, 0, 3).reshape(-1,4)

    return boxes


# def encode_boxes(boxes, anchors):
#     anchors_wh = anchors[:, 2:] - anchors[:, :2]
#     anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

#     boxes_wh = boxes[:, 2:] - boxes[:, :2]
#     boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh

#     encoded_xy = (boxes_ctr - anchors_ctr) / anchors_wh
#     encoded_wh = torch.log(boxes_wh / anchors_wh)

#     boxes = torch.cat([encoded_xy, encoded_wh], dim=1)

#     return boxes


# def compute_iou(src, dst):

#     p1 = torch.maximum(dst[:, None, :2], src[:, :2])
#     p2 = torch.minimum(dst[:, None, 2:], src[:, 2:])
    
#     inter = torch.prod((p2 - p1 + 1).clamp(min=0), dim=2)

#     src_area = torch.prod(src[:, 2:] - src[:, :2] + 1, dim=1)
#     dst_area = torch.prod(dst[:, 2:] - dst[:, :2] + 1, dim=1)

#     iou = torch.nan_to_num(inter / (dst_area[:, None] + src_area - inter))

#     return iou


# def decode_boxes(deltas, anchors):
#     anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1.
#     anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

#     pred_boxes_ctr = deltas[:,:2] * anchors_wh + anchors_ctr
#     pred_boxes_wh = deltas[:,2:].exp() * anchors_wh

#     pred_xmin_ymin = pred_boxes_ctr - 0.5*pred_boxes_wh
#     pred_xmax_ymax = pred_xmin_ymin + pred_boxes_wh -1.

#     return torch.cat([pred_xmin_ymin, pred_xmax_ymax], dim=1)



# def soft_nms(boxes, scores, sigma=0.5, score_thresh=0.001, topk=300):
#     # boxes: [N,4] xyxy, scores: [N]
#     keep = []
#     boxes = boxes.clone()
#     scores = scores.clone()
#     idxs = torch.arange(scores.numel(), device=scores.device)

#     out_boxes, out_scores = [], []
#     while scores.numel():
#         i = torch.argmax(scores)
#         bi, si, ii = boxes[i], scores[i], idxs[i]
#         out_boxes.append(bi); out_scores.append(si); keep.append(ii.item())

#         boxes = torch.cat((boxes[:i], boxes[i+1:]), 0)
#         scores = torch.cat((scores[:i], scores[i+1:]), 0)
#         idxs   = torch.cat((idxs[:i],   idxs[i+1:]),   0)

#         if scores.numel() == 0: break
#         # IoU with the picked box
#         x1 = torch.max(bi[0], boxes[:,0]); y1 = torch.max(bi[1], boxes[:,1])
#         x2 = torch.min(bi[2], boxes[:,2]); y2 = torch.min(bi[3], boxes[:,3])
#         inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
#         iou = inter / ((bi[2]-bi[0])*(bi[3]-bi[1]) + (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) - inter + 1e-9)
#         # Gaussian decay
#         scores = scores * torch.exp(- (iou * iou) / sigma)
#         # prune
#         keep_mask = scores >= score_thresh
#         boxes, scores, idxs = boxes[keep_mask], scores[keep_mask], idxs[keep_mask]

#     if len(keep) > topk:
#         order = torch.tensor(out_scores).argsort(descending=True)[:topk]
#         keep = [keep[i] for i in order.tolist()]
#     return keep


def compute_iou(src, dst):
    # src: [Ns,4], dst: [Nd,4]  (xyxy)
    lt = torch.maximum(dst[:, None, :2], src[:, :2])     # [Nd,Ns,2]
    rb = torch.minimum(dst[:, None, 2:], src[:, 2:])     # [Nd,Ns,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_s = (src[:, 2] - src[:, 0]).clamp(min=0) * (src[:, 3] - src[:, 1]).clamp(min=0)
    area_d = (dst[:, 2] - dst[:, 0]).clamp(min=0) * (dst[:, 3] - dst[:, 1]).clamp(min=0)
    union = area_d[:, None] + area_s - inter + 1e-9
    return inter / union

# SSD-style box coder w/o +1/-1 and with variances (next section):
VAR_CTR, VAR_SIZE = 0.1, 0.2

def encode_boxes(boxes, anchors):
    a_wh  = anchors[:, 2:] - anchors[:, :2]
    a_ctr = anchors[:, :2] + 0.5 * a_wh
    b_wh  = boxes[:, 2:] - boxes[:, :2]
    b_ctr = boxes[:, :2] + 0.5 * b_wh

    dxdy = (b_ctr - a_ctr) / (a_wh * VAR_CTR)
    dwdh = torch.log((b_wh / a_wh).clamp(min=1e-6)) / VAR_SIZE
    return torch.cat([dxdy, dwdh], dim=1)

def decode_boxes(deltas, anchors):
    a_wh  = anchors[:, 2:] - anchors[:, :2]
    a_ctr = anchors[:, :2] + 0.5 * a_wh

    dxy = deltas[:, :2] * VAR_CTR
    dwh = deltas[:, 2:] * VAR_SIZE
    dwh = dwh.clamp(min=-4.0, max=4.0).exp()

    p_ctr = a_ctr + dxy * a_wh
    p_wh  = a_wh * dwh
    half  = 0.5 * p_wh
    return torch.cat([p_ctr - half, p_ctr + half], dim=1)
