"""Géométrie du visage : détection MediaPipe FaceMesh et extraction des points.

Toutes les fonctions travaillent sur des tableaux NumPy (468, 2) en pixels,
pour que référence et vidéo soient directement comparables.
"""
from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


class DetecteurVisage:
    """Détecte un visage et retourne ses 468 points caractéristiques."""

    def __init__(self) -> None:
        self._photo = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self._video = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def fermer(self) -> None:
        self._photo.close()
        self._video.close()

    def points_photo(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Points du visage sur une image fixe (photo de référence)."""
        return self._points(image, self._photo)

    def points_video(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Points du visage sur une frame de vidéo (mode suivi)."""
        return self._points(image, self._video)

    def _points(self, image: np.ndarray, mesh) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        resultat = mesh.process(rgb)
        if not resultat.multi_face_landmarks:
            return None
        pts = resultat.multi_face_landmarks[0].landmark
        tableau = np.array(
            [[p.x * w, p.y * h] for p in pts], dtype=np.float32
        )
        return tableau
