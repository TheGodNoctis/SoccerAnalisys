
from typing import Dict, Any, List
import numpy as np

from team_assigner import TeamAssigner

# Paleta fixa (Time A / Time B)
PALETTE_HEX = {0: "#6EC1E4", 1: "#FF6FB5"}

def _hex_to_bgr(hx: str):
    hx = hx.lstrip('#')
    r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
    return (b, g, r)

def _apply_palette_fixed(tracks: Dict[str, Any], palette_hex=PALETTE_HEX):
    colors = {t: _hex_to_bgr(h) for t, h in palette_hex.items()}
    for frame_players in tracks.get('players', []):
        for pid, pdata in frame_players.items():
            t = int(pdata.get('team', 0))
            pdata['team_color'] = colors.get(t, colors[0])

def _stabilize_team_by_majority(tracks: Dict[str, Any]):
    votes = {}  # pid -> {team_id -> count}
    for frame_players in tracks.get('players', []):
        for pid, pdata in frame_players.items():
            t = pdata.get('team', None)
            if t is None:
                continue
            t = int(t)
            votes.setdefault(pid, {}).setdefault(t, 0)
            votes[pid][t] += 1

    final_team = {}
    for pid, counts in votes.items():
        final_team[pid] = max(counts.items(), key=lambda kv: kv[1])[0]

    for frame_players in tracks.get('players', []):
        for pid, pdata in frame_players.items():
            if pid in final_team:
                pdata['team'] = int(final_team[pid])

def assign_and_color_teams(video_frames: List[np.ndarray], tracks_players: List[Dict[int, Dict[str, Any]]], tracks_full: Dict[str, Any] | None = None):
    """
    Atribui time (0/1) com TeamAssigner, estabiliza por maioria por jogador
    e aplica a paleta fixa (#6EC1E4 / #FF6FB5).
    Modifica 'tracks_players' in place. Se 'tracks_full' for passado, aplica cor lá também.
    """
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks_players)

    # Fallback pontual para frames sem rótulo
    for frame_players in tracks_players:
        for pid, pdata in frame_players.items():
            if pdata.get('team') is None:
                t = team_assigner.get_player_team(video_frames[0], pdata['bbox'], pid)
                pdata['team'] = int(t) if t is not None else 0

    # Congela o time por maioria
    temp_tracks = {'players': tracks_players} if tracks_full is None else tracks_full
    _stabilize_team_by_majority(temp_tracks)

    # Aplica paleta fixa
    _apply_palette_fixed(temp_tracks)
