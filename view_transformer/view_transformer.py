import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_length = 120  # comprimento
        court_width = 70    # largura

        self.pixel_vertices = np.array([
            [110, 1035],  # inferior esquerdo
            [265, 275],   # superior esquerdo
            [910, 260],   # superior direito
            [1640, 915]   # inferior direito
        ], dtype=np.float32)
        
        self.target_vertices = np.array([
            [0, court_width],        # inferior esquerdo
            [0, 0],                  # superior esquerdo
            [court_length, 0],       # superior direito
            [court_length, court_width]  # inferior direito
        ], dtype=np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0:
            return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2).squeeze().tolist()

    def add_transformed_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, frame_track in enumerate(object_tracks):
                for track_id, track_info in frame_track.items():
                    position = track_info.get('position_adjusted')
                    if position is None:
                        continue
                    transformed = self.transform_point(np.array(position))
                    if transformed is not None:
                        tracks[object_name][frame_num][track_id]['position_transformed'] = transformed
                    else:
                        tracks[object_name][frame_num][track_id]['position_transformed'] = None
