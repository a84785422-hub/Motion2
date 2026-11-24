"""Transfert de mouvement : anime l'image de référence avec la vidéo source.

Principe (approche par maillage triangulé, standard en animation faciale) :
1. L'image de référence détecte 468 points du visage (MediaPipe FaceMesh).
2. Chaque frame de la vidéo détecte les mêmes 468 points.
3. On aligne d'abord rigoureusement la pose vidéo sur la pose de référence
   (échelle + rotation + translation, via similarité de Procruste).
4. Puis on amplifie uniquement le mouvement d'expression (bouche, sourcils,
   yeux) par rapport à la pose neutre de la référence.
5. Une triangulation de Delaunay découpe le visage en triangles ; chaque
   triangle subit une transformation affine référence -> cible.
6. Le visage déformé est fondu dans l'image d'origine via un masque flouté :
   fond, cheveux et épaules restent parfaitement intacts.
"""
from __future__ import annotations

import os
import subprocess
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from motion_engine.video_analysis import AnalyseVideo

Progression = Callable[[int, int], None]

# Points d'intérêt par région, indices FaceMesh
BOUCHE_HAUT, BOUCHE_BAS = 13, 14
COIN_G, COIN_D = 61, 291
SOURCIL_G, OEIL_G_HAUT = 105, 159
SOURCIL_D, OEIL_D_HAUT = 334, 386
OEIL_G_BAS, OEIL_D_BAS = 145, 374
PAUPIERE_G, PAUPIERE_D = 33, 263


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class MesuresVisage:
    """Mesures normalisées extraites d'un nuage de 468 points."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self.oeil_g = points[PAUPIERE_G]
        self.oeil_d = points[PAUPIERE_D]
        self.distance_oeils = max(distance(self.oeil_d, self.oeil_g), 1e-3)

    @property
    def ouverture_bouche(self) -> float:
        return distance(self.points[BOUCHE_BAS], self.points[BOUCHE_HAUT]) \
            / self.distance_oeils

    @property
    def largeur_bouche(self) -> float:
        return distance(self.points[COIN_D], self.points[COIN_G]) \
            / self.distance_oeils

    @property
    def elevation_sourcil_g(self) -> float:
        return distance(self.points[SOURCIL_G],
                        self.points[OEIL_G_HAUT]) / self.distance_oeils

    @property
    def elevation_sourcil_d(self) -> float:
        return distance(self.points[SOURCIL_D],
                        self.points[OEIL_D_HAUT]) / self.distance_oeils

    @property
    def ouverture_oeil_g(self) -> float:
        return distance(self.points[OEIL_G_HAUT],
                        self.points[OEIL_G_BAS]) / self.distance_oeils

    @property
    def ouverture_oeil_d(self) -> float:
        return distance(self.points[OEIL_D_HAUT],
                        self.points[OEIL_D_BAS]) / self.distance_oeils

    def vecteur_expression(self) -> np.ndarray:
        """Vecteur de 6 mesures d'expression, normalisées par les yeux."""
        return np.array([
            self.ouverture_bouche,
            self.largeur_bouche,
            self.elevation_sourcil_g,
            self.elevation_sourcil_d,
            self.ouverture_oeil_g,
            self.ouverture_oeil_d,
        ], dtype=np.float64)


class AligneurPose:
    """Aligne chaque pose vidéo sur la pose de référence (Procruste 2D)."""

    def __init__(self, points_reference: np.ndarray) -> None:
        self.points_reference = points_reference.astype(np.float64)
        self.mesures_reference = MesuresVisage(points_reference)
        self.centre_reference = points_reference.mean(axis=0)
        self.echelle_reference = self._echelle(points_reference)
        self.locale_reference = self._vers_local(points_reference)

    def _echelle(self, points: np.ndarray) -> float:
        """Échelle = distance entre les coins externes des yeux."""
        return max(distance(points[PAUPIERE_D], points[PAUPIERE_G]), 1e-3)

    def _vers_local(self, points: np.ndarray) -> np.ndarray:
        """Points centrés et mis à l'échelle (invariants au cadrage)."""
        centre = points.mean(axis=0)
        return (points - centre) / self._echelle(points)

    def aligner(self, points_video: np.ndarray,
                amplification: float) -> np.ndarray:
        """Retourne les points cibles dans le repère de l'image de référence.

        amplification > 1 accentue l'expression, 1 = fidèle à la vidéo,
        < 1 atténue.
        """
        locale_video = self._vers_local(points_video)
        locale_ref = self.locale_reference

        # rotation optimale entre les deux nuages (Procruste 2D)
        matrice_cov = locale_video.T @ locale_ref
        u, _, vt = np.linalg.svd(matrice_cov)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt

        aligned = locale_video @ rotation

        # expression : on n'amplifie que l'écart entre les deux formes
        if amplification != 1.0:
            moyenne_video = aligned.mean(axis=0)
            ecart = aligned - moyenne_video
            aligned = moyenne_video + ecart * amplification

        # retour dans le repère de l'image de référence
        return aligned * self.echelle_reference + self.centre_reference


def trianguler(points: np.ndarray) -> np.ndarray:
    """Triangulation de Delaunay sur le visage de référence."""
    x, y, l, h = cv2.boundingRect(points.astype(np.int32))
    subdiv = cv2.Subdiv2D((x, y, l, h))
    subdiv.insert([(float(p[0]), float(p[1])) for p in points])
    triangles = subdiv.getTriangleList()

    resultat = []
    for t in triangles:
        idx = []
        for sx, sy in ((t[0], t[1]), (t[2], t[3]), (t[4], t[5])):
            dist = np.linalg.norm(points - np.array([sx, sy]), axis=1)
            proche = int(np.argmin(dist))
            if dist[proche] < 2.0:
                idx.append(proche)
        if len(idx) == 3 and len(set(idx)) == 3:
            resultat.append(idx)
    return np.array(resultat, dtype=np.int32)


def masque_visage(forme: Tuple[int, int], points: np.ndarray,
                  marge: float = 0.06, flou: int = 41) -> np.ndarray:
    """Masque du visage (convex hull) avec marge et bord fondu."""
    h, l = forme
    centre = points.mean(axis=0)
    gonfle = centre + (points - centre) * (1.0 + marge)
    masque = np.zeros((h, l), dtype=np.uint8)
    cv2.fillConvexPoly(masque, gonfle.astype(np.int32), 255)
    return cv2.GaussianBlur(masque, (flou, flou), 0)


def deformer_visage(image_reference: np.ndarray,
                    points_reference: np.ndarray,
                    points_cible: np.ndarray,
                    triangles: np.ndarray) -> np.ndarray:
    """Warp triangle par triangle de la référence vers les points cibles."""
    sortie = np.zeros_like(image_reference)
    ref = points_reference.astype(np.float32)
    cbl = points_cible.astype(np.float32)

    for i1, i2, i3 in triangles:
        src = np.array([ref[i1], ref[i2], ref[i3]], dtype=np.float32)
        dst = np.array([cbl[i1], cbl[i2], cbl[i3]], dtype=np.float32)

        r1 = cv2.boundingRect(src.astype(np.int32))
        r2 = cv2.boundingRect(dst.astype(np.int32))
        if r1[2] == 0 or r1[3] == 0 or r2[2] == 0 or r2[3] == 0:
            continue

        src_roi = src - np.array([r1[0], r1[1]], dtype=np.float32)
        dst_roi = dst - np.array([r2[0], r2[1]], dtype=np.float32)

        matrice = cv2.getAffineTransform(src_roi, dst_roi)
        morceau = image_reference[r1[1]:r1[1] + r1[3],
                                  r1[0]:r1[0] + r1[2]]
        morceau_warp = cv2.warpAffine(
            morceau, matrice, (r2[2], r2[3]), None,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101,
        )

        masque_tri = np.zeros((r2[3], r2[2]), dtype=np.uint8)
        cv2.fillConvexPoly(masque_tri, np.int32(dst_roi), 255, 16, 0)

        roi = sortie[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
        masque_bool = masque_tri.astype(bool)[:, :, None]
        sortie[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = np.where(
            masque_bool, morceau_warp, roi
        )
    return sortie


def generer_video(image_reference: np.ndarray,
                  points_reference: np.ndarray,
                  analyse: AnalyseVideo,
                  chemin_sortie: str,
                  amplification: float = 1.15,
                  lissage: int = 2,
                  limite_frames: int = 900,
                  marge_masque: float = 0.06,
                  callback: Optional[Progression] = None) -> str:
    """Génère la vidéo animée complète.

    - image_reference : photo de départ (BGR).
    - points_reference : 468 points du visage de la photo.
    - analyse : poses extraites de la vidéo de mouvement.
    - amplification : 1.0 = fidèle, 1.3 = plus expressif.
    - lissage : nombre de frames moyennées (2 = léger, 5 = très lisse).

    Retourne le chemin de la vidéo produite.
    """
    if analyse.nb_frames_valides == 0:
        raise ValueError("La vidéo ne contient aucun visage exploitable.")

    os.makedirs(os.path.dirname(os.path.abspath(chemin_sortie)),
                exist_ok=True)

    aligneur = AligneurPose(points_reference)
    triangles = trianguler(points_reference)
    if len(triangles) == 0:
        raise ValueError("Triangulation du visage impossible : "
                         "utilisez une photo plus nette et mieux cadrée.")

    analyse.lisser(lissage)
    if analyse.nb_frames > limite_frames:
        analyse.tronquer(limite_frames)

    hauteur, largeur = image_reference.shape[:2]
    redaction = cv2.VideoWriter(
        chemin_sortie,
        cv2.VideoWriter_fourcc(*"mp4v"),
        analyse.fps or 30.0,
        (largeur, hauteur),
    )
    if not redaction.isOpened():
        raise RuntimeError("Impossible de créer le fichier vidéo de sortie.")

    masque = masque_visage(image_reference.shape[:2], points_reference,
                           marge=marge_masque)
    masque_norm = (masque.astype(np.float32) / 255.0)[:, :, None]
    fond = image_reference.astype(np.float32) * (1.0 - masque_norm)

    reference_neutre = MesuresVisage(points_reference).vecteur_expression()

    try:
        total = analyse.nb_frames
        for i in range(total):
            points_video = analyse.points[i]
            if points_video is None:
                redaction.write(image_reference)
                continue

            cibles = aligneur.aligner(points_video, amplification)

            # régulation de l'expression : on ajuste les mesures d'expression
            # de la cible vers celles de la vidéo (sinon la mise à l'échelle
            # de Procruste écrase les expressions fines)
            cibles = _reguler_expression(points_reference, cibles,
                                         points_video, aligneur,
                                         reference_neutre, amplification)

            visage = deformer_visage(image_reference, points_reference,
                                     cibles, triangles)
            compose = fond + visage.astype(np.float32) * masque_norm
            redaction.write(compose.astype(np.uint8))

            if callback and i % 5 == 0:
                callback(i + 1, total)
    finally:
        redaction.release()

    _transcoder_h264(chemin_sortie)
    return chemin_sortie


def _reguler_expression(points_reference: np.ndarray,
                        points_cible: np.ndarray,
                        points_video: np.ndarray,
                        aligneur: AligneurPose,
                        reference_neutre: np.ndarray,
                        amplification: float) -> np.ndarray:
    """Ajuste finement bouche / sourcils / yeux de la cible sur la vidéo.

    Le warp Procruste préserve la forme globale, mais les déplacements fins
    (ouverture de bouche, sourcils levés) doivent être recalés explicitement
    sur les mesures de la vidéo source.
    """
    mesures_video = MesuresVisage(points_video).vecteur_expression()
    mesures_cible = MesuresVisage(points_cible).vecteur_expression()

    # écart vidéo vs référence, multiplié par l'amplification
    delta = (mesures_video - reference_neutre) * amplification
    correctif = delta - (mesures_cible - reference_neutre)

    resultat = points_cible.copy()
    echelle = aligneur.echelle_reference

    def deplace(i: int, dy: float) -> None:
        resultat[i, 1] += dy

    def deplace_x(i: int, dx: float) -> None:
        resultat[i, 0] += dx

    # Bouche verticale (ouverture)
    dy_bouche = correctif[0] * echelle * 0.5
    deplace(BOUCHE_BAS, dy_bouche)
    deplace(BOUCHE_HAUT, -dy_bouche)

    # Largeur de bouche (sourire)
    dx_bouche = correctif[1] * echelle * 0.5
    deplace_x(COIN_G, -dx_bouche)
    deplace_x(COIN_D, dx_bouche)

    # Sourcils
    dy_sourcil_g = correctif[2] * echelle * 0.5
    dy_sourcil_d = correctif[3] * echelle * 0.5
    for i in (95, 96, 105, 107, 52, 51, 50):
        deplace(i, dy_sourcil_g)
    for i in (324, 325, 334, 336, 282, 281, 280):
        deplace(i, dy_sourcil_d)

    # Yeux (clignement)
    dy_oeil_g = correctif[4] * echelle * 0.5
    dy_oeil_d = correctif[5] * echelle * 0.5
    deplace(OEIL_G_BAS, dy_oeil_g)
    deplace(OEIL_G_HAUT, -dy_oeil_g)
    deplace(OEIL_D_BAS, dy_oeil_d)
    deplace(OEIL_D_HAUT, -dy_oeil_d)

    return resultat


def _transcoder_h264(chemin: str) -> None:
    """Ré-encode en H.264 avec ffmpeg s'il est présent (compatibilité
    maximale des lecteurs, taille réduite). Sinon, conserve mp4v."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return

    tmp = chemin + ".h264.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", chemin,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", tmp],
            capture_output=True, check=True,
        )
        os.replace(tmp, chemin)
    except (subprocess.CalledProcessError, OSError):
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
