"""Moteur Motion Control : transfert de mouvement d'une vidéo vers une image."""
from motion_engine.face_analysis import DetecteurVisage
from motion_engine.video_analysis import AnalyseVideo
from motion_engine.motion_transfer import generer_video

__all__ = ["DetecteurVisage", "AnalyseVideo", "generer_video"]
