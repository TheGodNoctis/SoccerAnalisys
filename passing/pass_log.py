def log_passes(tracks, fps=24, min_hold_sec=1.2, min_pass_duration_sec=1.8):
    min_hold_frames = int(fps * min_hold_sec)
    min_pass_duration_frames = int(fps * min_pass_duration_sec)

    passes = []
    last_player_id = None
    last_team = None
    possession_start_frame = 0

    # Variáveis para posse provisória (para ignorar mudanças rápidas)
    temp_player_id = None
    temp_team = None
    temp_start_frame = 0

    for frame_num, players in enumerate(tracks['players']):
        current_player_id = None
        current_team = None

        # Detecta jogador com bola no frame atual
        for player_id, player in players.items():
            if player.get("has_ball", False):
                current_player_id = player_id
                current_team = player.get("team", None)
                break

        if current_player_id is None:
            continue

        if last_player_id is None:
            # Primeira posse detectada
            last_player_id = current_player_id
            last_team = current_team
            possession_start_frame = frame_num
            continue

        if current_player_id == last_player_id:
            # Mesmo jogador mantém a posse
            temp_player_id = None
            temp_team = None
            continue

        if current_team == last_team:
            # Passa para jogador do mesmo time - só registra se durou tempo mínimo
            duration = frame_num - possession_start_frame
            if duration >= min_pass_duration_frames:
                passes.append({
                    "from_id": last_player_id,
                    "to_id": current_player_id,
                    "team": current_team,  # time que fez o passe
                    "start_frame": possession_start_frame,
                    "end_frame": frame_num,
                    "pass_type": "normal"
                })
            # Atualiza posse
            last_player_id = current_player_id
            possession_start_frame = frame_num
            temp_player_id = None
            temp_team = None
        else:
            # Mudou para adversário - usa posse provisória para confirmar interceptação
            if temp_player_id is None:
                temp_player_id = current_player_id
                temp_team = current_team
                temp_start_frame = frame_num
            else:
                if current_player_id != temp_player_id:
                    # Mudou para outro adversário diferente - reseta temporário
                    temp_player_id = current_player_id
                    temp_team = current_team
                    temp_start_frame = frame_num
                else:
                    duration = frame_num - temp_start_frame
                    if duration >= min_hold_frames:
                        # Posse adversária confirmada -> passe interceptado
                        passes.append({
                            "from_id": last_player_id,
                            "to_id": current_player_id,
                            "team": last_team,  # time que perdeu a posse (quem fez o passe interceptado)
                            "start_frame": possession_start_frame,
                            "end_frame": frame_num,
                            "pass_type": "interceptado"
                        })
                        # Atualiza posse principal para adversário confirmado
                        last_player_id = current_player_id
                        last_team = current_team
                        possession_start_frame = frame_num
                        temp_player_id = None
                        temp_team = None

    return passes
