# minimap_utils.py — utilitários para desenho do mini-campo
import numpy as np
import cv2
import supervision as sv
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration

# Configuração de campo
CONFIG_PITCH = SoccerPitchConfiguration()
VERTS = np.array(CONFIG_PITCH.vertices, dtype=np.float32)
FIELD_UNITS_LENGTH = float(VERTS[:, 0].max())
FIELD_UNITS_WIDTH  = float(VERTS[:, 1].max())

# Estilo padrão
COLOR_YELLOW = (0, 215, 255)
COLOR_BLACK  = (0, 0, 0)

def _hex_to_bgr(hx: str):
    hx = hx.lstrip('#')
    r = int(hx[0:2], 16)
    g = int(hx[2:4], 16)
    b = int(hx[4:6], 16)
    return (b, g, r)

def _bgr(c, fallback=None):
    try:
        if c is None:
            return fallback
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

def _frame_to_pitch_xy(points_xy, H_fp):
    if H_fp is None or points_xy is None or len(points_xy) == 0:
        return None
    pts = np.asarray(points_xy, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_fp).reshape(-1, 2)

def _build_full_pitch(target_w):
    pitch_img = draw_pitch(CONFIG_PITCH)
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex('#FFFFFF'),
        thickness=2,
        edges=CONFIG_PITCH.edges
    )
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
        color = _bgr(pdata.get("team_color"), None)
        centers.append(p)
        metas.append((color, pdata.get("has_ball", False)))

    if H_fp is not None and len(centers) > 0:
        pts_m = _frame_to_pitch_xy(centers, H_fp)
        pts_m[:, 0] = np.clip(pts_m[:, 0], 0, FIELD_UNITS_LENGTH)
        pts_m[:, 1] = np.clip(pts_m[:, 1], 0, FIELD_UNITS_WIDTH)
        pad = 10
        sx = (w_dst - 2 * pad) / FIELD_UNITS_LENGTH
        sy = (h_dst - 2 * pad) / FIELD_UNITS_WIDTH
        pts_px = np.stack(
            [pad + pts_m[:, 0] * sx, pad + pts_m[:, 1] * sy],
            axis=1
        ).astype(np.int32)
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
            sx = (w_dst - 2 * pad) / FIELD_UNITS_LENGTH
            sy = (h_dst - 2 * pad) / FIELD_UNITS_WIDTH
            bpx, bpy = int(pad + bm[0, 0] * sx), int(pad + bm[0, 1] * sy)
        else:
            bpx, bpy = int(bx / w_src * w_dst), int(by / h_src * h_dst)
        cv2.circle(pitch_img, (bpx, bpy), 6, (255, 255, 255), -1)
        cv2.circle(pitch_img, (bpx, bpy), 6, (0, 0, 0), 1)
