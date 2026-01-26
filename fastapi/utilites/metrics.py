import torch
import torch.nn.functional as F
import numpy as np



def calculate_expansion_flow(keypoints, sensitivity = 0.5):
    keypoints = keypoints[:, : -4, :]
    velocities = keypoints[1:] - keypoints[:-1]
    centers = keypoints[:-1].mean(dim=1, keepdim=True)
    radial = keypoints[:-1] - centers
    avg_radius = torch.norm(radial, dim=2).mean() + 1e-6
    radial_flow = (radial * velocities).sum(dim=2)
    expansion_score = radial_flow.mean() / avg_radius
    
    return (torch.tanh(0.15 * expansion_score) + 1)
def get_mean_gaze_direction_v3(keypoints, scores, focus_point=(320, 320), 
                                attention_radius=15, orientation_weight = 0.8, power=2):
    device = keypoints.device
    dtype = keypoints.dtype

    left_ear = keypoints[:, 14, :]
    right_ear = keypoints[:, 15, :]
    nose = keypoints[:, 16, :]
    focus = torch.tensor(focus_point, dtype=dtype, device=device).unsqueeze(0)

    head_center = (left_ear + right_ear) / 2
    gaze_dir = nose - head_center
    gaze_dir = gaze_dir / (torch.norm(gaze_dir, dim=1, keepdim=True) + 1e-6)

    to_focus = focus - nose
    to_focus = to_focus / (torch.norm(to_focus, dim=1, keepdim=True) + 1e-6)

    alignment = (gaze_dir * to_focus).sum(dim=1).clamp(-1, 1)
    alignment_score = (alignment + 1) / 2  # map [-1,1] → [0,1]
    dist = torch.norm(focus - nose, dim=1)
    print(f"Distace: {dist}")
    dist_norm = 1 - torch.clamp(dist / 200, 0, 1)  # 0 if far, 1 if close
    score = orientation_weight * alignment_score + (1 - orientation_weight) * dist_norm
    print(alignment_score)
    return score.mean()

def camera_facing_v2(keypoints, scores):
    """
    Returns scalar in [0,1]: 1 = facing camera, 0 = facing away.
    Penalizes side-facing and back-facing poses.
    """
    left_ear_conf = scores[:, 14].unsqueeze(1)
    right_ear_conf = scores[:, 15].unsqueeze(1)
    nose_conf = scores[:, 16].unsqueeze(1)
    mean_conf = (left_ear_conf + right_ear_conf + 2 * nose_conf) / 4
    symmetry = 1 - torch.abs(left_ear_conf - right_ear_conf)
    away_penalty = 1 - torch.clamp((mean_conf * 2), 0, 1)  # 1→strong penalty when all low
    facing_raw = mean_conf * (0.7 * symmetry + 0.3 * (1 - away_penalty))
    facing_score = torch.sigmoid(10 * (facing_raw.mean() - 0.5))
    return facing_score

    

def get_average_head_tilt_angle(keypoints, scores=None, conf_threshold=0.1, max_angle = 20.0):
    left_ear = keypoints[:, 14, :]
    right_ear = keypoints[:, 15, :]
    
    if scores is not None:
        conf_left = scores[:, 14]
        conf_right = scores[:, 15]
        valid_mask = (conf_left > conf_threshold) & (conf_right > conf_threshold)
        if valid_mask.sum() == 0:
            0.0
        left_ear = left_ear[valid_mask]
        right_ear = right_ear[valid_mask]
        weights = ((conf_left[valid_mask] + conf_right[valid_mask]) / 2)
    else:
        weights = None

    dx = right_ear[:, 0] - left_ear[:, 0]
    dy = right_ear[:, 1] - left_ear[:, 1]
    tilt_rad = torch.atan2(torch.abs(dy), torch.abs(dx))
    tilt_deg = torch.rad2deg(tilt_rad)

    if weights is not None:
        avg_tilt_deg = (tilt_deg * weights).sum() / (weights.sum() + 1e-6)
    else:
        avg_tilt_deg = tilt_deg.mean()
    return torch.clamp(avg_tilt_deg / max_angle, 0.0, 1.0)

def get_bbox_trend(bboxes, window_size = 5, sensitivity = 0.5):
    if len(bboxes) < window_size:
        return 0.0, {'status': 'insufficient_frames'}
    bboxes = torch.tensor(bboxes).to("cuda").squeeze()
    w, h = bboxes[:, 2], bboxes[:, 3]
    areas = w * h
    if len(areas) >= 5:
        kernel = min(5, len(areas))
        areas_smooth = torch.nn.functional.avg_pool1d(
            areas.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=kernel//2
        ).squeeze()[:len(areas)]
    else:
        areas_smooth = areas
    diffs = areas_smooth[1:] - areas_smooth[:-1]
    if len(diffs) > 0:
        bbox_change_rate = diffs.mean() / (areas_smooth.mean() + 1e-6)
    else:
        bbox_change_rate = torch.tensor(0.0, device=areas.device)
    return torch.sigmoid(10 * bbox_change_rate)

def get_intent_score(xy, scores, bboxes):
    gaze = get_mean_gaze_direction_v3(xy, scores)
    head_tilt = get_average_head_tilt_angle(xy, scores)
    bbox_flow_rate = get_bbox_trend(bboxes)
    radial_flow_rate = calculate_expansion_flow(xy)
    camera_facing_score  = camera_facing_v2(xy, scores)
    print(camera_facing_score, gaze, head_tilt, bbox_flow_rate, radial_flow_rate)
    intent_score = 0.5 * camera_facing_score + 0.15 * gaze + 0.1 * head_tilt + 0.15 * bbox_flow_rate + 0.1 * radial_flow_rate
    return intent_score
