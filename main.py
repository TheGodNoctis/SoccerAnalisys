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

def main():
    # 1️⃣ Leitura do vídeo
    video_path = 'input_videos/573e61_0.mp4'
    video_frames = read_video(video_path)
    print(f"[INFO] Vídeo carregado: {len(video_frames)} frames")

    # 2️⃣ Inicializa Tracker
    tracker = Tracker('models/bestx.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    # 3️⃣ Estimador de movimento de câmera
    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(video_frames, read_from_stub=True,
                                                                     stub_path='stubs/camera_movement_stub.pkl')
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # 4️⃣ Transformação de coordenadas para campo 2D
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # 5️⃣ Interpolação da bola
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 6️⃣ Atribuição de times
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks['players'])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # 7️⃣ Posse de bola
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

    # 8️⃣ Velocidade e distância
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)
    print("[INFO] Velocidade e distância calculadas.")

    # 9️⃣ Passes e estatísticas
    passes = log_passes(tracks, fps=24)
    stats = summarize_passes(passes)

    # 1️⃣0️⃣ Geração de relatório
    os.makedirs("output_reports", exist_ok=True)
    report_text = generate_report(tracks, passes, stats, team_ball_control)
    with open('output_reports/analise_tatica.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[INFO] Relatório salvo.")

    # 1️⃣1️⃣ Renderização do vídeo
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)
    os.makedirs("output_videos", exist_ok=True)
    save_video(output_frames, 'output_videos/testeDistance.avi')
    print("[INFO] Vídeo exportado com sucesso.")

if __name__ == "__main__":
    main()
