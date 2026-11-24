"""Analyse d'une vidéo de référence : extraction des points du visage frame par frame.

Les 468 points de chaque frame sont mémorisés dans un fichier cache (.npz)
afin de ne jamais ré-analyser deux fois la même vidéo.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional

import cv2
import numpy as np

from motion_engine.face_analysis import DetecteurVisage

NB_POINTS = 468


class AnalyseVideo:
    """Résultat de l'analyse frame par frame d'une vidéo."""

    def __init__(self) -> None:
        self.points: List[Optional[np.ndarray]] = []
        self.nb_frames: int = 0
        self.fps: float = 30.0
        self.taille: tuple = (0, 0)  # (largeur, hauteur)

    @property
    def nb_frames_valides(self) -> int:
        return sum(1 for p in self.points if p is not None)

    def analyser(self, chemin_video: str, detecteur: DetecteurVisage,
                 callback_progression: Optional[Callable[[int, int], None]] = None
                 ) -> None:
        """Analyse la vidéo complète. Lève ValueError si aucun visage détecté."""
        capture = cv2.VideoCapture(chemin_video)
        if not capture.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {chemin_video}")

        self.fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self.taille = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.points = []
        total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

        idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # réduction de charge : analyse en demi-résolution suffisant
            petit = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            pts = detecteur.points_video(petit)
            if pts is not None:
                pts = pts * 2.0  # retour à l'échelle réelle
            self.points.append(pts)
            idx += 1
            if callback_progression and idx % 5 == 0:
                callback_progression(idx, total)
        capture.release()

        self.nb_frames = len(self.points)
        if self.nb_frames_valides == 0:
            raise ValueError(
                "Aucun visage détecté dans la vidéo. Utilisez une vidéo où "
                "le visage est bien visible, de face et bien éclairé."
            )

    # ------------------------------------------------------------- traitement
    def lisser(self, demi_fenetre: int) -> None:
        """Moyenne glissante sur les points pour réduire le tremblement.

        Les trous (frames sans visage) sont comblés par interpolation entre
        les frames valides voisines, pour une animation sans à-coups.
        """
        if self.nb_frames == 0:
            return
        self._combler_trous()
        if demi_fenetre <= 0:
            return
        tableaux = np.stack([p for p in self.points if p is not None])
        n = len(tableaux)
        lisser = tableaux.copy()
        for i in range(n):
            debut = max(0, i - demi_fenetre)
            fin = min(n, i + demi_fenetre + 1)
            lisser[i] = tableaux[debut:fin].mean(axis=0)
        self.points = [lisser[i] for i in range(n)]

    def _combler_trous(self) -> None:
        """Interpole les frames sans visage détecté entre deux frames valides."""
        n = len(self.points)
        if self.points[0] is None:
            premier = next((i for i in range(n) if self.points[i] is not None),
                           None)
            if premier is not None:
                for i in range(premier):
                    self.points[i] = self.points[premier]
        if self.points[n - 1] is None:
            dernier = next((i for i in range(n - 1, -1, -1)
                            if self.points[i] is not None), None)
            if dernier is not None:
                for i in range(dernier + 1, n):
                    self.points[i] = self.points[dernier]

        i = 0
        while i < n:
            if self.points[i] is not None:
                i += 1
                continue
            j = i
            while j < n and self.points[j] is None:
                j += 1
            if j >= n:
                break
            avant = self.points[i - 1]
            apres = self.points[j]
            for k in range(i, j):
                t = (k - i + 1) / (j - i + 1)
                self.points[k] = avant * (1 - t) + apres * t
            i = j

    def tronquer(self, nb_max: int) -> None:
        if self.nb_frames > nb_max:
            self.points = self.points[:nb_max]
            self.nb_frames = nb_max

    # ------------------------------------------------------------- persistance
    def sauvegarder(self, chemin: str) -> None:
        valides = np.array([p is not None for p in self.points], dtype=np.int8)
        tableaux = np.zeros((self.nb_frames, NB_POINTS, 2), dtype=np.float32)
        for i, p in enumerate(self.points):
            if p is not None:
                tableaux[i] = p
        np.savez_compressed(
            chemin,
            tableaux=tableaux,
            valides=valides,
            fps=self.fps,
            taille=np.array(self.taille),
        )

    @staticmethod
    def charger(chemin: str) -> "AnalyseVideo":
        data = np.load(chemin)
        analyse = AnalyseVideo()
        analyse.fps = float(data["fps"])
        analyse.taille = tuple(int(v) for v in data["taille"])
        valides = data["valides"]
        analyse.nb_frames = len(valides)
        analyse.points = [
            data["tableaux"][i] if valides[i] else None
            for i in range(analyse.nb_frames)
        ]
        return analyse

    @staticmethod
    def chemin_cache(chemin_video: str) -> str:
        base, _ = os.path.splitext(chemin_video)
        return base + "_motioncache.npz"

    @staticmethod
    def charger_ou_analyser(chemin_video: str, detecteur: DetecteurVisage,
                            callback_progression:
                            Optional[Callable[[int, int], None]] = None
                            ) -> "AnalyseVideo":
        """Utilise le cache existant s'il est plus récent que la vidéo."""
        cache = AnalyseVideo.chemin_cache(chemin_video)
        if (os.path.exists(cache)
                and os.path.getmtime(cache) >= os.path.getmtime(chemin_video)):
            try:
                return AnalyseVideo.charger(cache)
            except Exception:
                pass  # cache corrompu : on ré-analyse
        analyse = AnalyseVideo()
        analyse.analyser(chemin_video, detecteur, callback_progression)
        try:
            analyse.sauvegarder(cache)
        except OSError:
            pass
        return analyse
