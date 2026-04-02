"""3D Non-Maximum Suppression for bounding box filtering."""

import torch


def box_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned BEV IoU between two sets of boxes.

    Args:
        boxes_a: (N, 7) — x, y, z, w, l, h, yaw
        boxes_b: (M, 7) — x, y, z, w, l, h, yaw

    Returns:
        IoU matrix (N, M)
    """
    # Use axis-aligned approximation for BEV IoU (ignoring yaw for speed)
    x1_a = boxes_a[:, 0] - boxes_a[:, 3] / 2
    y1_a = boxes_a[:, 1] - boxes_a[:, 4] / 2
    x2_a = boxes_a[:, 0] + boxes_a[:, 3] / 2
    y2_a = boxes_a[:, 1] + boxes_a[:, 4] / 2

    x1_b = boxes_b[:, 0] - boxes_b[:, 3] / 2
    y1_b = boxes_b[:, 1] - boxes_b[:, 4] / 2
    x2_b = boxes_b[:, 0] + boxes_b[:, 3] / 2
    y2_b = boxes_b[:, 1] + boxes_b[:, 4] / 2

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    inter_x1 = torch.max(x1_a.unsqueeze(1), x1_b.unsqueeze(0))
    inter_y1 = torch.max(y1_a.unsqueeze(1), y1_b.unsqueeze(0))
    inter_x2 = torch.min(x2_a.unsqueeze(1), x2_b.unsqueeze(0))
    inter_y2 = torch.min(y2_a.unsqueeze(1), y2_b.unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area

    return inter_area / (union_area + 1e-8)


def nms_3d(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5
) -> torch.Tensor:
    """Greedy 3D NMS using BEV IoU.

    Args:
        boxes: (N, 7) bounding boxes
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: (K,) indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)

        if order.numel() == 1:
            break

        current_box = boxes[idx].unsqueeze(0)
        remaining_boxes = boxes[order[1:]]
        iou = box_iou_bev(current_box, remaining_boxes).squeeze(0)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
