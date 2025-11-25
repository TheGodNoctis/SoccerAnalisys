from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
import torch 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
        self.id_correct = {
            #37: 11,   # sempre tratar 37 como 11
            # 58: 4,  # se tiver outros casos, coloca aqui
            102: 2,
            66: 2,
            68: 11,
            73: 22,
            101:14,
            57: 14,
            78: 14,
            90: 14,
            48: 14,
            100: 17,
            106: 9,
            129: 21
        }        
        
        self.fixed_team = {
            23:2,
            14: 2,
            21: 1,
            19: 1,
        }

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_tracks, max_speed=60):
        num_frames = len(ball_tracks)
        centers = []

        # 1. Extrair centros
        for frame in ball_tracks:
            ball_info = frame.get(1, {}).get('bbox', [])
            if ball_info:
                x1, y1, x2, y2 = ball_info
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                centers.append([x_center, y_center, width, height])
            else:
                centers.append([np.nan, np.nan, np.nan, np.nan])

        # 2. Criar DataFrame
        df = pd.DataFrame(centers, columns=["x", "y", "w", "h"])

        # 3. Interpolação linear
        df = df.interpolate(method="linear", limit_direction="both")
        df = df.ffill().bfill()

        # 4. Remover outliers de velocidade com suavização progressiva
        for i in range(1, len(df)):
            dx = abs(df.loc[i, "x"] - df.loc[i-1, "x"])
            dy = abs(df.loc[i, "y"] - df.loc[i-1, "y"])
            if dx > max_speed or dy > max_speed:
                # não trava seco → suaviza entre anterior e atual
                df.loc[i, "x"] = 0.7 * df.loc[i-1, "x"] + 0.3 * df.loc[i, "x"]
                df.loc[i, "y"] = 0.7 * df.loc[i-1, "y"] + 0.3 * df.loc[i, "y"]

        # 5. Aplicar suavização exponencial (EMA)
        df["x"] = df["x"].ewm(span=5, adjust=False).mean()
        df["y"] = df["y"].ewm(span=5, adjust=False).mean()

        # 6. Reconstruir bounding boxes
        interpolated_ball_tracks = []
        for i in range(num_frames):
            x_center, y_center, w, h = df.iloc[i]
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            bbox = [x1, y1, x2, y2]
            interpolated_ball_tracks.append({1: {"bbox": bbox}})

        return interpolated_ball_tracks


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, device=self.device)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    canonical_id = self.id_correct.get(track_id, track_id)
                    tracks["players"][frame_num][canonical_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                                    

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = y2 - rectangle_height // 2 + 15
            y2_rect = y2 + rectangle_height // 2 + 15

            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_1_num_frames = np.sum(team_ball_control[:frame_num + 1] == 1)
        team_2_num_frames = np.sum(team_ball_control[:frame_num + 1] == 2)

        total = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total if total > 0 else 0
        team_2 = team_2_num_frames / total if total > 0 else 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for canonical_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, canonical_id)

                #if player.get('has_ball', False):
                    #frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            #for _, ball in ball_dict.items():
                #frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            #frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames

