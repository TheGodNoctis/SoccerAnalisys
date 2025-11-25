import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance, get_foot_position

class PlayerBallAssigner():
    def __init__(self, fps=24, min_hold_sec=0.5):
        self.max_player_ball_distance = 50  
        self.fps = fps
        self.min_hold_frames = int(fps * min_hold_sec)

        self.current_player = None     # Jogador atualmente com posse confirmada
        self.current_team = None       # Time do jogador com posse confirmada
        self.temp_player = None        # Jogador provisório
        self.temp_team = None
        self.temp_start_frame = None   # Frame de início da posse provisória
        self.frame_counter = 0         # Contador de frames

    def assign_ball_to_player(self, players, ball_bbox):
        self.frame_counter += 1
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = float("inf")
        candidate_player = None
        candidate_team = None

        for player_id, player in players.items():
            player_id = int(player_id)
            player_bbox = player['bbox']
            foot_pos = get_foot_position(player_bbox)  # usa o centro do pé
            distance = measure_distance(foot_pos, ball_position)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                candidate_player = player_id
                candidate_team = player.get("team", None)

        if candidate_player is None:
            self.current_player = None
            self.current_team = None
            self.temp_player = None
            self.temp_team = None
            self.temp_start_frame = None
            return None

        if self.current_player is None:
            self.current_player = candidate_player
            self.current_team = candidate_team
            return self.current_player

        if candidate_player == self.current_player:
            self.temp_player = None
            self.temp_team = None
            self.temp_start_frame = None
            return self.current_player

        if self.temp_player is None or candidate_player != self.temp_player:
            self.temp_player = candidate_player
            self.temp_team = candidate_team
            self.temp_start_frame = self.frame_counter
            return self.current_player  # mantém posse anterior até confirmar

        else:
            duration = self.frame_counter - self.temp_start_frame
            if duration >= self.min_hold_frames:
                self.current_player = candidate_player
                self.current_team = candidate_team
                self.temp_player = None
                self.temp_team = None
                self.temp_start_frame = None
                return self.current_player
            else:
                return self.current_player
