from utils import read_video, save_video
from trackers import Tracker
import numpy as np
import os
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from passing.pass_log import log_passes
from passing.pass_stats import summarize_passes
from generate_report import generate_report
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class AnnotationOptions:
    player_boxes: bool = True
    player_ids: bool = True
    team_colors: bool = True
    ball: bool = True
    possession: bool = False
    speed_distance: bool = True
    pass_events: bool = True
    trails: bool = False
    field_transform: bool = False
    mini_pitch: bool = True            

import cv2, time, pickle
from ultralytics import YOLO
import supervision as sv

from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
CONFIG_PITCH = SoccerPitchConfiguration()
VERTS = np.array(CONFIG_PITCH.vertices, dtype=np.float32)
FIELD_UNITS_LENGTH = float(VERTS[:, 0].max())  # ex.: 12000.0
FIELD_UNITS_WIDTH  = float(VERTS[:, 1].max())  # ex.: 7000.0

MINI_PITCH_ALPHA_DEFAULT = 0.5
MINI_PITCH_MARGIN_DEFAULT = 12
COLOR_YELLOW = (0, 215, 255)
COLOR_BLACK  = (0, 0, 0)
DEFAULT_FIELD_MODEL_PATH = Path(os.getenv("FIELD_MODEL_PATH", "best (2).pt"))

PALETTE_HEX = {0: "#6EC1E4", 1: "#FF6FB5"}  # Team A (0) / Team B (1)

def _hex_to_bgr(hx: str):
    hx = hx.lstrip('#')
    r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
    return (b, g, r)

def _apply_palette_fixed(tracks, palette_hex=PALETTE_HEX):
    colors = {t: _hex_to_bgr(h) for t, h in palette_hex.items()}
    for frame_players in tracks.get('players', []):
        for pid, pdata in frame_players.items():
            t = int(pdata.get('team', 0))
            pdata['team_color'] = colors.get(t, colors[0])


def hex_to_bgr(hx: str):
    hx = hx.lstrip('#')
    r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
    return (b, g, r)

def _bgr(c, fallback=None):
    try:
        if c is None: return fallback
        c = np.asarray(c).astype(float).flatten()
        if c.size >= 3:
            return (int(c[0]), int(c[1]), int(c[2]))
    except Exception:
        pass
    return fallback

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def ground_point_for_player(pdata):
    if "bbox" in pdata:
        x1, y1, x2, y2 = pdata["bbox"]
        return ((x1 + x2) / 2.0, y2)
    return None

def overlay_bottom_center(base_img, overlay_img, alpha=1.0, bottom_margin=12):
    h, w = base_img.shape[:2]
    oh, ow = overlay_img.shape[:2]
    x = (w - ow) // 2
    y = max(0, h - oh - bottom_margin)
    if y + oh > h or x + ow > w:
        oh = min(oh, h - bottom_margin); ow = min(ow, w)
        overlay_img = cv2.resize(overlay_img, (ow, oh))
    roi = base_img[y:y+oh, x:x+ow]
    if alpha < 1.0:
        blended = cv2.addWeighted(roi, 1.0 - alpha, overlay_img, alpha, 0.0)
        base_img[y:y+oh, x:x+ow] = blended
    else:
        base_img[y:y+oh, x:x+ow] = overlay_img
    return base_img

def _infer_field_keypoints(frame, field_model, conf_filter=0.55):
    result = field_model(frame, verbose=False)[0]
    if hasattr(result, "keypoints") and result.keypoints is not None:
        kxy = result.keypoints.xy[0].cpu().numpy()
        kcf = result.keypoints.conf[0].cpu().numpy()
        mask = kcf > conf_filter
        pf = kxy[mask].astype(np.float32)
        pp = np.array(CONFIG_PITCH.vertices)[mask].astype(np.float32)
        if len(pf) >= 4:
            return pf, pp
    return None, None

def _estimate_H_with_refit(pf, pp, reproj_th=2.0):
    H, mask = cv2.findHomography(pp, pf, cv2.RANSAC, reproj_th)
    if H is None: return None
    inl = mask.ravel().astype(bool)
    pf_in, pp_in = pf[inl], pp[inl]
    if len(pf_in) >= 4:
        H_refit, _ = cv2.findHomography(pp_in, pf_in, 0)
        if H_refit is not None:
            return H_refit
    return H

def _blend_H(H_prev, H_new, alpha=0.25):
    if H_prev is None: return H_new
    return (1 - alpha) * H_prev + alpha * H_new

def _compute_homographies(video_frames, field_model, stride=30, conf_filter=0.55, smooth_alpha=0.25):
    H_list, Hpf_prev = [], None
    for i, f in enumerate(video_frames):
        cur = None
        if (i == 0) or (i % stride == 0):
            pf, pp = _infer_field_keypoints(f, field_model, conf_filter=conf_filter)
            if pf is not None:
                Hpf_tmp = _estimate_H_with_refit(pf, pp)
                if Hpf_tmp is not None:
                    Hpf_prev = _blend_H(Hpf_prev, Hpf_tmp, alpha=smooth_alpha)
        if Hpf_prev is not None:
            cur = {"H_pf": Hpf_prev, "H_fp": np.linalg.inv(Hpf_prev)}
        H_list.append(cur)
    return H_list

def _frame_to_pitch_xy(points_xy, H_fp):
    if H_fp is None or points_xy is None or len(points_xy) == 0:
        return None
    pts = np.asarray(points_xy, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_fp).reshape(-1, 2)

def _build_full_pitch(target_w):
    pitch_img = draw_pitch(CONFIG_PITCH)
    edge_annotator = sv.EdgeAnnotator(color=sv.Color.from_hex('#FFFFFF'), thickness=2, edges=CONFIG_PITCH.edges)
    keypoints = sv.KeyPoints(xy=np.array(CONFIG_PITCH.vertices)[np.newaxis, ...])
    pitch_img = edge_annotator.annotate(scene=pitch_img, key_points=keypoints)
    target_h = int(pitch_img.shape[0] * target_w / pitch_img.shape[1])
    pitch_img = cv2.resize(pitch_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return pitch_img

def _draw_minimap_on_pitch(pitch_img, frame_size, players_dict, ball_bbox, H_fp):
    h_src, w_src = frame_size
    h_dst, w_dst = pitch_img.shape[:2]
    centers, metas = [], []

    for pid, pdata in players_dict.items():
        p = ground_point_for_player(pdata)
        if p is None: 
            continue
        team_id = int(pdata.get("team", 0))
        color = _bgr(pdata.get("team_color"), None)
        centers.append(p)
        metas.append((color, pdata.get("has_ball", False)))

    if H_fp is not None and len(centers) > 0:
        pts_m = _frame_to_pitch_xy(centers, H_fp)
        pts_m[:, 0] = np.clip(pts_m[:, 0], 0, FIELD_UNITS_LENGTH)
        pts_m[:, 1] = np.clip(pts_m[:, 1], 0, FIELD_UNITS_WIDTH)
        pad = 10
        sx = (w_dst - 2*pad) / FIELD_UNITS_LENGTH
        sy = (h_dst - 2*pad) / FIELD_UNITS_WIDTH
        pts_px = np.stack([pad + pts_m[:, 0] * sx, pad + pts_m[:, 1] * sy], axis=1).astype(np.int32)
    else:
        pts_px = np.array([[int(x), int(y)] for (x, y) in centers], dtype=np.int32)

    DOT_RADIUS, DOT_OUTLINE = 7, 2
    for (px, py), (color, has_ball) in zip(pts_px, metas):
        cv2.circle(pitch_img, (px, py), DOT_RADIUS + DOT_OUTLINE, COLOR_BLACK, -1)
        cv2.circle(pitch_img, (px, py), DOT_RADIUS, color if color else (255, 255, 255), -1)
        if has_ball:
            cv2.circle(pitch_img, (px, py), DOT_RADIUS + 3, COLOR_YELLOW, 2)

    if ball_bbox is not None:
        bx, by = bbox_center(ball_bbox)
        if H_fp is not None:
            bm = _frame_to_pitch_xy([[bx, by]], H_fp)
            pad = 10
            sx = (w_dst - 2*pad) / FIELD_UNITS_LENGTH
            sy = (h_dst - 2*pad) / FIELD_UNITS_WIDTH
            bpx, bpy = int(pad + bm[0, 0] * sx), int(pad + bm[0, 1] * sy)
        else:
            bpx, bpy = int(bx / w_src * w_dst), int(by / h_src * h_dst)
        cv2.circle(pitch_img, (bpx, bpy), 6, (255, 255, 255), -1)
        cv2.circle(pitch_img, (bpx, bpy), 6, (0, 0, 0), 1)

def process_video(
    video_path: str,
    use_stubs: bool = True,
    output_dir: str = "outputs",
    fps: int = 24,
    ann: Optional[AnnotationOptions | Dict[str, Any]] = None,
    mini_pitch_scale: float = 0.25,
    mini_pitch_alpha: float = MINI_PITCH_ALPHA_DEFAULT,
    mini_pitch_margin: int = MINI_PITCH_MARGIN_DEFAULT,
    field_conf_filter: float = 0.55,
    field_stride: int = 30,
    field_smooth_alpha: float = 0.25,
) -> dict:
    # Normaliza opções
    if ann is None:
        ann = AnnotationOptions()
    elif isinstance(ann, dict):
        ann = AnnotationOptions(**ann)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = Path(output_dir) / f"-{ts}"
    vid_out_dir = job_dir / "output_videos"
    rep_out_dir = job_dir / "output_reports"
    os.makedirs(vid_out_dir, exist_ok=True)
    os.makedirs(rep_out_dir, exist_ok=True)

    video_frames = read_video(video_path)

    tracker = Tracker('models/bestx.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path='stubs/track_stubs.pkl'
    )
    tracker.add_position_to_tracks(tracks)

    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(
        video_frames,
        read_from_stub=use_stubs,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks['players'])
    for frame_players in tracks['players']:
        for pid, pdata in frame_players.items():
            if pdata.get('team') is None:
                t = team_assigner.get_player_team(video_frames[0], pdata['bbox'], pid)
                pdata['team'] = int(t) if t is not None else 0

    _apply_palette_fixed(tracks)   

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1 and assigned_player in tracks['players'][frame_num]:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) > 0 else -1)
    team_ball_control = np.array(team_ball_control)

    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    passes = log_passes(tracks, fps=fps)
    stats = summarize_passes(passes)

    report_text = generate_report(tracks, passes, stats, team_ball_control)
    report_path = rep_out_dir / 'analise_tatica.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    output_frames = video_frames
    try:
        output_frames = tracker.draw_annotations(
            video_frames, tracks, team_ball_control, options=asdict(ann)
        )
    except TypeError:
        output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    if ann.mini_pitch:
        H_list = None
        field_model = None

        if DEFAULT_FIELD_MODEL_PATH.exists():
            try:
                field_model = YOLO(str(DEFAULT_FIELD_MODEL_PATH))
                H_list = _compute_homographies(
                    video_frames,
                    field_model,
                    stride=field_stride,
                    conf_filter=field_conf_filter,
                    smooth_alpha=field_smooth_alpha,
                )
            except Exception as e:
                print(f"[mini-pitch] Aviso: falha ao usar modelo de campo ({DEFAULT_FIELD_MODEL_PATH}): {e}")
        else:
            print(f"[mini-pitch] Modelo de campo não encontrado em {DEFAULT_FIELD_MODEL_PATH}. Usando modo básico (sem homografia).")

        for i in range(len(output_frames)):
            frame_h, frame_w = output_frames[i].shape[:2]
            pitch_w = max(80, int(frame_w * mini_pitch_scale))
            pitch_img = _build_full_pitch(pitch_w)
            H_fp = None
            if H_list and H_list[i] is not None:
                H_fp = H_list[i]["H_fp"]

            try:
                ball_bbox = tracks["ball"][i][1]["bbox"]
            except Exception:
                ball_bbox = None

            _draw_minimap_on_pitch(
                pitch_img,
                (frame_h, frame_w),
                tracks["players"][i],
                ball_bbox,
                H_fp
            )

            output_frames[i] = overlay_bottom_center(
                output_frames[i],
                pitch_img,
                alpha=mini_pitch_alpha,
                bottom_margin=mini_pitch_margin,
            )

    if ann.speed_distance:
        output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)

    video_out_path = vid_out_dir / 'video_processado.avi'
    save_video(output_frames, str(video_out_path))

    return {
        'report_path': str(report_path),
        'video_path': str(video_out_path),
        'job_dir': str(job_dir),
        'stats': stats,
        'annotation_options': asdict(ann),
    }
