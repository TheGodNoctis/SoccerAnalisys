import numpy as np
import cv2
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator:
    def __init__(self, frame_window=24, frame_rate=24):
        self.frame_window = frame_window
        self.frame_rate = frame_rate
        

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object_name, object_tracks in tracks.items():
            if object_name in ["ball", "referees"]:
                continue

            total_distance[object_name] = {}

            num_frames = len(object_tracks)
            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_pos = object_tracks[frame_num][track_id].get('position_transformed')
                    end_pos = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_pos is None or end_pos is None:
                        continue

                    # Distância em metros (original)
                    distance_covered = (measure_distance(start_pos, end_pos)) * 103

                    # Tempo em segundos
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    if time_elapsed <= 0:
                        continue

                    # Velocidade em m/s
                    speed_mps = (distance_covered / time_elapsed) / 288

                    total_distance[object_name][track_id] = total_distance[object_name].get(track_id, 0) + distance_covered
                    total_distance_div = total_distance[object_name][track_id] / 1000  # ajuste

                    # Atualiza todos os frames da janela
                    for fn in range(frame_num, last_frame + 1):
                        if track_id not in object_tracks[fn]:
                            continue
                        object_tracks[fn][track_id]['speed'] = speed_mps
                        object_tracks[fn][track_id]['distance'] = total_distance_div 

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object_name, object_tracks in tracks.items():
                if object_name in ["ball", "referees"]:
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    speed = track_info.get('speed')
                    distance = track_info.get('distance')
                    if speed is None or distance is None:
                        continue

                    bbox = track_info['bbox']
                    position = get_foot_position(bbox)
                    position = list(position)
                    position[1] += 40
                    position = tuple(map(int, position))

                    cv2.putText(frame, f"{speed:.2f} m/s", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            output_frames.append(frame)

        return output_frames

