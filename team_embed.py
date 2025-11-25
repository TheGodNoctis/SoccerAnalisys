# team_embed.py — Classificação de times por embeddings (SigLIP via sports.TeamClassifier)
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import supervision as sv  # pip install supervision>=0.23
from sports.common.team import TeamClassifier  # pip install git+https://github.com/roboflow/sports.git

# Paleta fixa (A/B) que você já usa no projeto
PALETTE_HEX = {0: "#6EC1E4", 1: "#FF6FB5"}

def _hex_to_bgr(hx: str):
    hx = hx.lstrip('#'); r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
    return (b, g, r)

def _apply_palette_fixed(tracks: Dict[str, Any], palette_hex=PALETTE_HEX):
    colors = {t: _hex_to_bgr(h) for t, h in palette_hex.items()}
    for frame_players in tracks.get('players', []):
        for pid, pdata in frame_players.items():
            t = int(pdata.get('team', 0))
            pdata['team_color'] = colors.get(t, colors[0])

def _stabilize_team_by_majority(tracks: Dict[str, Any]):
    votes = {}
    for frame_players in tracks.get('players', []):
        for pid, pdata in frame_players.items():
            t = pdata.get('team', None)
            if t is None: 
                continue
            t = int(t)
            votes.setdefault(pid, {}).setdefault(t, 0)
            votes[pid][t] += 1
    final_team = {pid: max(counts.items(), key=lambda kv: kv[1])[0] for pid, counts in votes.items()}
    for frame_players in tracks.get('players', []):
        for pid, pdata in frame_players.items():
            if pid in final_team:
                pdata['team'] = int(final_team[pid])

def _crop(frame: np.ndarray, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1: 
        return None
    return frame[y1:y2, x1:x2]

def fit_team_classifier(video_frames: List[np.ndarray], tracks_players: List[Dict[int, Dict[str, Any]]], stride: int = 30):
    """
    Coleta crops de jogadores (um a cada 'stride' frames), e treina o TeamClassifier.
    """
    crops = []
    for i in range(0, len(video_frames), max(1, int(stride))):
        frame = video_frames[i]
        if i >= len(tracks_players): 
            break
        for _, pdata in tracks_players[i].items():
            roi = _crop(frame, pdata["bbox"])
            if roi is not None:
                crops.append(roi)
    if len(crops) < 2:
        return None  # sem dados suficientes
    # O notebook faz exatamente isso: criar o classificador e dar fit nos crops :contentReference[oaicite:2]{index=2}
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TeamClassifier(device=device)
    # TeamClassifier aceita PIL Images; converter rápido via util do supervision
    crops_pil = [sv.cv2_to_pillow(c) for c in crops]
    clf.fit(crops_pil)
    return clf

def assign_teams_with_embeddings(video_frames: List[np.ndarray], tracks: Dict[str, Any], stride: int = 30):
    """
    Treina o classificador de times e rotula cada jogador em cada frame com 0/1.
    Depois estabiliza por maioria e aplica paleta fixa.
    """
    players = tracks['players']
    clf = fit_team_classifier(video_frames, players, stride=stride)
    if clf is None:
        # fallback: marca todo mundo como time 0 e aplica paleta
        for frame_players in players:
            for _, pdata in frame_players.items():
                pdata['team'] = 0
        _apply_palette_fixed(tracks)
        return

    # Predizer por frame: crops -> predict -> class_id (0/1)
    # O exemplo mostra .predict(crops) nos jogadores do frame :contentReference[oaicite:3]{index=3}
    for i, frame_players in enumerate(players):
        frame = video_frames[i]
        batch = []
        pids = []
        for pid, pdata in frame_players.items():
            roi = _crop(frame, pdata["bbox"])
            if roi is None: 
                continue
            batch.append(sv.cv2_to_pillow(roi))
            pids.append(pid)
        if len(batch) == 0:
            continue
        preds = clf.predict(batch)  # retorna 0/1
        for pid, t in zip(pids, preds):
            frame_players[pid]['team'] = int(t)

    # Estabiliza por maioria e aplica a paleta azul/rosa
    _stabilize_team_by_majority(tracks)
    _apply_palette_fixed(tracks)
