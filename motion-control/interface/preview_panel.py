"""Panneau de visualisation : affiche les images/vidéos dans l'interface."""
from __future__ import annotations

import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np


class PanneauApercu(ctk.CTkFrame):
    """Affiche une image redimensionnée avec bandeau d'état."""

    TAILLE_MAX = 520

    def __init__(self, parent, titre: str, **kwargs) -> None:
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._photo = None  # référence Tk obligatoire
        self.titre = ctk.CTkLabel(
            self, text=titre, font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#1a1a1a", "#e8f0ec"),
        )
        self.titre.pack(pady=(0, 6))

        self.cadre = ctk.CTkFrame(
            self, corner_radius=12,
            fg_color=("#eceff0", "#16211c"),
            border_width=1, border_color=("#d0d6d3", "#2a3a33"),
        )
        self.cadre.pack(fill="both", expand=True)

        self.etiquette = ctk.CTkLabel(
            self.cadre,
            text="Aucun fichier choisi",
            text_color=("#7a8a83", "#8fa69c"),
            font=ctk.CTkFont(size=13),
        )
        self.etiquette.place(relx=0.5, rely=0.5, anchor="center")

    def afficher_image(self, image: np.ndarray, legende: str = "") -> None:
        """Affiche une image OpenCV (BGR) ajustée au panneau."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, l = rgb.shape[:2]
        echelle = min(self.TAILLE_MAX / max(h, 1),
                      self.TAILLE_MAX / max(l, 1), 1.0)
        taille = (max(int(l * echelle), 1), max(int(h * echelle), 1))
        rgb = cv2.resize(rgb, taille, interpolation=cv2.INTER_AREA)

        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photo = photo

        self.etiquette.configure(text=legende or " ", image=photo,
                                 compound="top", text_color=("#5a6a63",
                                                             "#8fa69c"))

    def afficher_message(self, message: str, couleur: str = "gray") -> None:
        self.etiquette.configure(image=None, text=message, text_color=couleur)
        self._photo = None
