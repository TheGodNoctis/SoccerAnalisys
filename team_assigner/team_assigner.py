from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        h = image.shape[0]
        if h < 2:
            return np.array([0, 0, 0])
        shirt = image[:h // 2, :]
        shorts = image[h // 2:, :]
        shirt_color = shirt.reshape(-1, 3).mean(axis=0)
        shorts_color = shorts.reshape(-1, 3).mean(axis=0)
        combined_color = (0.7 * shirt_color + 0.3 * shorts_color)
        return combined_color

    def assign_team_color(self, frames, player_tracks, sample_every_n_frames=10):
        player_colors = []
        for i in range(0, len(frames), sample_every_n_frames):
            frame = frames[i]
            for player_id, track in player_tracks[i].items():
                bbox = track["bbox"]
                color = self.get_player_color(frame, bbox)
                player_colors.append(color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        self.player_team_dict[player_id] = team_id
        return team_id



