import requests
from requests_cache import CachedSession
import os
from urllib.parse import quote
import numpy as np
import time
import sys
import json

VERBOSE = False

class TrackInfo:
    def __init__(self, name: str= None, artist: str= None, album: str= None, year: int= None, id: str = None, href: str = None, releaseDate: str = None) -> None:
        self.name = name
        self.artist = artist
        self.album = album
        self.year = year
        self.id = id
        self.href = href
        if releaseDate:
            self.year = releaseDate[-4:]

    def genQuery(self) -> str:
        qName = "track:"+self.name if self.name else None
        qArtist = "artist:"+self.artist if self.artist else None
        qAlbum = "album:"+self.album if self.album else None
        qYear = "year:"+str(self.year) if self.year else None
        q = [qName, qArtist, qAlbum, qYear]
        q = " ".join([x for x in q if x])
        return quote(q)

class AudioFeatures:
    def __init__(self, acousticness: float= None, danceability: float= None, duration_ms: int= None, energy: float= None, instrumentalness: float= None, key: int= None, liveness: float= None, loudness: float= None, 
        mode: int= None, speechiness: float= None, tempo: float= None, time_signature: int= None, valence: float= None, url: str= None, type: str = None, id: str = None, uri: str = None, track_href: str =None, analysis_url:str =None) -> None:
        self.duration_ms = duration_ms
        self.key = key
        self.mode = mode
        self.tempo = tempo
        self.time_signature = time_signature
        self.url = url
        self.type = type
        self.id = id
        self.uri = uri
        self.track_href = track_href
        self.analysis_url = analysis_url
        self.model = AudioModel(acousticness, danceability, energy, instrumentalness, liveness, speechiness, valence, loudness)

class AudioModel:
    def __init__(self, acousticness: float= None, danceability: float= None, energy: float= None, instrumentalness: float= None, liveness: float= None, speechiness: float= None, valence: float= None, loudness: float= None) -> None:
        self.acousticness = acousticness
        self.danceability = danceability
        self.energy = energy
        self.instrumentalness = instrumentalness
        self.liveness = liveness
        self.speechiness = speechiness
        self.valence = valence
        self.loudness = loudness

    def getNumpyVector(self) -> np.ndarray:
        return np.array([self.acousticness, self.danceability, self.energy, self.instrumentalness, self.liveness, self.speechiness, self.valence, self.loudness])

class Track:
    def __init__(self, audio_features: AudioFeatures = None, track_info: TrackInfo = None) -> None:
        self.audio_features = audio_features
        self.track_info = track_info

class PlaylistBuilder:
    def __init__(self, client_id: str, client_secret: str, tracks: list[Track]) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.tracks = tracks
        self.auth = self.getAuthtoken(client_id, client_secret)
        self.session = CachedSession(cache_name="cache", backend="sqlite", expire_after=3600)

    def run(self, limit: int):
        tracks = [self.searchTrack(track) for track in self.tracks]
        tracks = [track for track in tracks if track]
        tracks = self.getAudioFeatures(tracks)
        model = self.genAverageModel(tracks)
        seeds = self.getBestSeeds(tracks, model)
        recommendedSongs = self.getModelRecommendations(model, seeds, limit=limit)
        # Remove duplicates
        recommendedSongs = [track for track in recommendedSongs if track.track_info.id not in [track.track_info.id for track in tracks]]
        if len(recommendedSongs) > limit:
            recommendedSongs = recommendedSongs[:limit]
        # Make playlist a set to remove duplicates
        return recommendedSongs

    def getAuthtoken(self, client_id: str, client_secret: str) -> str:
        if (VERBOSE): print("GENERATING AUTH TOKEN")
        response = requests.post("https://accounts.spotify.com/api/token", data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret})
        return "Bearer " + response.json()["access_token"]

    def searchTrack(self, track: Track):
        if (VERBOSE): print("SEARCHING FOR TRACK: " + track.track_info.name)
        q = track.track_info.genQuery()
        response = self.session.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": self.auth, "Content-Type": "application/json", "Accept": "application/json"}, 
            params={"q": q, "type": "track", "limit": 1}
        )
        if not response.from_cache:
            time.sleep(SLEEP_TIME)
        if len(response.json()["tracks"]["items"]) == 0:
            return
        track = response.json()["tracks"]["items"][0]
        data = {
            "name": track["name"],
            "artist": track["artists"][0]["name"],
            "album": track["album"]["name"],
            "year": track["album"]["release_date"][:4],
            "id": track["id"],
            "href": track["href"]
        }
        track = Track(track_info=TrackInfo(**data))
        return track

    def getAudioFeatures(self, tracks: list[Track]):
        if (VERBOSE): print("GETTING AUDIO FEATURES")
        ids = [track.track_info.id for track in tracks]
        response = self.session.get(
            "https://api.spotify.com/v1/audio-features",
            headers={"Authorization": self.auth, "Content-Type": "application/json", "Accept": "application/json"},
            params={"ids": ",".join(ids)}
        )
        if not response.from_cache:
            time.sleep(SLEEP_TIME)
        features = response.json()["audio_features"]
        for track, feature in zip(tracks, features):
            track.audio_features = AudioFeatures(**feature)
        return tracks

    def genAverageModel(self, tracks: list[Track]) -> AudioModel:
        if (VERBOSE): print("GENERATING AVERAGE MODEL")
        mat = np.matrix([track.audio_features.model.getNumpyVector() for track in tracks])
        return AudioModel(*mat.mean(axis=0).tolist()[0])

    def getBestSeeds(self,tracks: list[Track], model: AudioModel, limit: int = 5) -> list[Track]:
        if (VERBOSE): print("GETTING BEST SEEDS")
        # if len(tracks) <= limit:
        #     return tracks
        mat = np.matrix([track.audio_features.model.getNumpyVector() for track in tracks])
        dist = np.linalg.norm(mat - model.getNumpyVector(), axis=1)
        return [tracks[i] for i in dist.argsort()[:limit]]

    def getModelRecommendations(self, model: AudioModel, seed_tracks: list[Track], limit: int = 10, cache: bool = True):
        if (VERBOSE): print("GENERATING RECOMMENDATIONS")
        # Todo: replace with top 5 most similar tracks to model
        ids = [track.track_info.id for track in seed_tracks]
        params = {
                    "seed_tracks": ",".join(ids),
                    "limit": limit,
                    "target_acousticness": model.acousticness,
                    "target_danceability": model.danceability,
                    "target_energy": model.energy,
                    "target_instrumentalness": model.instrumentalness,
                    "target_liveness": model.liveness,
                    "target_speechiness": model.speechiness,
                    "target_valence": model.valence,
                    "target_loudness": model.loudness
                }
        if cache:
            response = self.session.get(
                "https://api.spotify.com/v1/recommendations",
                headers={"Authorization": self.auth, "Content-Type": "application/json", "Accept": "application/json"},
                params=params
            )
        else:
            with self.session.cache_disabled():
                response = self.session.get(
                    "https://api.spotify.com/v1/recommendations",
                    headers={"Authorization": self.auth, "Content-Type": "application/json", "Accept": "application/json"},
                    params=params
                )
        if not response.from_cache:
            time.sleep(SLEEP_TIME)
        tracks = []
        for track in response.json()["tracks"]:
            data = {
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "album": track["album"]["name"],
                "year": track["album"]["release_date"][:4],
                "id": track["id"],
                "href": track["href"]
            }
            track = Track(track_info=TrackInfo(**data))
            tracks.append(track)
        return tracks


if __name__ == "__main__":
    CLIENT_ID = os.environ.get("CLIENT_ID")
    CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
    SLEEP_TIME = 0.1

    data = " ".join(sys.argv[1:])
    data = json.loads(data)

    tracks = [Track(track_info=TrackInfo(**track)) for track in data["tracks"]]
    # TODO: Cache token

    playlistBuilder = PlaylistBuilder(CLIENT_ID, CLIENT_SECRET, tracks)

    playlist = playlistBuilder.run(data["limit"])

    playlist = [track.track_info.__dict__ for track in playlist]

    playlist = {
        "tracks": playlist,
    }

    print(json.dumps(playlist))

    

    # TODO: Fix inserting repeated tracks

# WITH TARGETS
# クラウディ Simon & Garfunkel
# Carried Away Crosby, Stills & Nash
# After All Al Jarreau
# Just Once Quincy Jones
# (They Long To Be) Close To You Carpenters
# Fire and Rain - 2019 Remaster James Taylor
# Through the Eyes of Love (Theme from the Motion Picture "Ice Castles") Melissa Manchester
# Cathedral Crosby, Stills & Nash
# In Your Eyes George Benson
# April Come She Will Simon & Garfunkel

# WITHOUT TARGETS
# I Am a Rock Simon & Garfunkel
# If You Leave Me Now Chicago
# Poor Shirley Christopher Cross
# Is This Love - 2018 Remaster Whitesnake
# If Bread
# I Just Can't Let Go Ambrosia
# No More Lonely Nights (Ballad) - Remastered 1993 Paul McCartney
# How Long (feat. Christopher Cross) Jeff Golub with Brian Auger
# Sweet Baby James - 2019 Remaster James Taylor
# Early Mornin' Rain Gordon Lightfoot