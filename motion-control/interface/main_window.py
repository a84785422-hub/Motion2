"""Motion Control — Animation faciale par transfert de mouvement.

Interface principale : on choisit une image de référence, une vidéo de
mouvement, et le programme produit une vidéo où l'image de référence
reproduit les mouvements du visage de la vidéo.
"""
from __future__ import annotations

import os
import threading
import traceback
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np

from motion_engine.constants import (
    COULEUR_ACCENT,
    COULEUR_ACCENT_SOMBRE,
    COULEUR_ALERTE,
    COULEUR_ERREUR,
    COULEUR_FOND,
    COULEUR_PANNEAU,
    COULEUR_TEXTE,
    COULEUR_TEXTE_SECONDAIRE,
    FORMATS_IMAGE,
    FORMATS_VIDEO,
)
from motion_engine.face_analysis import DetecteurVisage
from motion_engine.motion_transfer import generer_video
from motion_engine.video_analysis import AnalyseVideo
from interface.preview_panel import PanneauApercu


class Application(ctk.CTk):
    """Fenêtre principale du programme."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.title("Motion Control — Animation faciale par transfert de mouvement")
        self.geometry("1180x760")
        self.minsize(980, 680)
        self.configure(fg_color=COULEUR_FOND)

        # état
        self.detecteur = DetecteurVisage()
        self.chemin_image: Optional[str] = None
        self.chemin_video: Optional[str] = None
        self.image_reference: Optional[np.ndarray] = None
        self.points_reference: Optional[np.ndarray] = None
        self.analyse: Optional[AnalyseVideo] = None
        self.en_cours = False
        self.chemin_sortie: Optional[str] = None

        self._construire_interface()
        self.protocol("WM_DELETE_WINDOW", self._quitter)

    # ------------------------------------------------------------------ UI
    def _construire_interface(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._construire_entete()
        self._construire_panneau_gauche()
        self._construire_panneau_droit()
        self._construire_bandeau_controles()

    def _construire_entete(self) -> None:
        entete = ctk.CTkFrame(self, fg_color="transparent", height=70)
        entete.grid(row=0, column=0, columnspan=2, sticky="ew",
                    padx=24, pady=(20, 8))
        entete.grid_propagate(False)

        ctk.CTkLabel(
            entete, text="Motion Control",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=COULEUR_ACCENT,
        ).pack(side="left")

        ctk.CTkLabel(
            entete,
            text="Anime une image de référence avec les mouvements d'une vidéo",
            font=ctk.CTkFont(size=13),
            text_color=COULEUR_TEXTE_SECONDAIRE,
        ).pack(side="left", padx=18, pady=(8, 0))

        self.badge_etat = ctk.CTkLabel(
            entete, text="Prêt", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COULEUR_PANNEAU, corner_radius=8,
            text_color=COULEUR_ACCENT, padx=12, pady=4,
        )
        self.badge_etat.pack(side="right", pady=(6, 0))

    def _construire_panneau_gauche(self) -> None:
        gauche = ctk.CTkFrame(self, fg_color="transparent")
        gauche.grid(row=1, column=0, sticky="nsew", padx=(24, 8), pady=8)

        self.apercu_image = PanneauApercu(gauche, "1 · Image de référence")
        self.apercu_image.pack(fill="both", expand=True)

        ctk.CTkButton(
            gauche, text="Choisir une image…", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COULEUR_ACCENT_SOMBRE, hover_color=COULEUR_ACCENT,
            text_color="#04231a",
            command=self._choisir_image,
        ).pack(fill="x", pady=(8, 0))

        self.etat_image = ctk.CTkLabel(
            gauche, text="Aucune image", anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=COULEUR_TEXTE_SECONDAIRE,
        )
        self.etat_image.pack(fill="x", pady=(4, 0))

    def _construire_panneau_droit(self) -> None:
        droit = ctk.CTkFrame(self, fg_color="transparent")
        droit.grid(row=1, column=1, sticky="nsew", padx=(8, 24), pady=8)

        self.apercu_video = PanneauApercu(droit, "2 · Vidéo de mouvement")
        self.apercu_video.pack(fill="both", expand=True)

        rangee = ctk.CTkFrame(droit, fg_color="transparent")
        rangee.pack(fill="x", pady=(8, 0))
        rangee.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            rangee, text="Choisir une vidéo…", height=40,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COULEUR_ACCENT_SOMBRE, hover_color=COULEUR_ACCENT,
            text_color="#04231a",
            command=self._choisir_video,
        ).pack(fill="x")

        self.etat_video = ctk.CTkLabel(
            rangee, text="Aucune vidéo", anchor="w",
            font=ctk.CTkFont(size=11),
            text_color=COULEUR_TEXTE_SECONDAIRE,
        )
        self.etat_video.pack(fill="x", pady=(4, 0))

    def _construire_bandeau_controles(self) -> None:
        bandeau = ctk.CTkFrame(self, fg_color=COULEUR_PANNEAU,
                               corner_radius=14)
        bandeau.grid(row=2, column=0, columnspan=2, sticky="ew",
                     padx=24, pady=(8, 22))
        bandeau.grid_columnconfigure((0, 1, 2), weight=1)

        # --- réglage : intensité de l'expression
        bloc_expr = ctk.CTkFrame(bandeau, fg_color="transparent")
        bloc_expr.grid(row=0, column=0, sticky="ew", padx=18, pady=14)
        ctk.CTkLabel(bloc_expr, text="Intensité de l'expression",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COULEUR_TEXTE).pack(anchor="w")
        self.slider_expression = ctk.CTkSlider(
            bloc_expr, from_=0.5, to=2.0, number_of_steps=30,
            command=self._maj_etiquettes,
        )
        self.slider_expression.set(1.15)
        self.slider_expression.pack(fill="x", pady=(6, 0))
        self.val_expression = ctk.CTkLabel(
            bloc_expr, text="1.15", font=ctk.CTkFont(size=11),
            text_color=COULEUR_TEXTE_SECONDAIRE,
        )
        self.val_expression.pack(anchor="e")

        # --- réglage : lissage
        bloc_liss = ctk.CTkFrame(bandeau, fg_color="transparent")
        bloc_liss.grid(row=0, column=1, sticky="ew", padx=18, pady=14)
        ctk.CTkLabel(bloc_liss, text="Lissage du mouvement",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COULEUR_TEXTE).pack(anchor="w")
        self.slider_lissage = ctk.CTkSlider(
            bloc_liss, from_=0, to=6, number_of_steps=6,
            command=self._maj_etiquettes,
        )
        self.slider_lissage.set(2)
        self.slider_lissage.pack(fill="x", pady=(6, 0))
        self.val_lissage = ctk.CTkLabel(
            bloc_liss, text="2", font=ctk.CTkFont(size=11),
            text_color=COULEUR_TEXTE_SECONDAIRE,
        )
        self.val_lissage.pack(anchor="e")

        # --- réglage : durée maximale
        bloc_duree = ctk.CTkFrame(bandeau, fg_color="transparent")
        bloc_duree.grid(row=0, column=2, sticky="ew", padx=18, pady=14)
        ctk.CTkLabel(bloc_duree, text="Durée maximale (secondes)",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COULEUR_TEXTE).pack(anchor="w")
        self.slider_duree = ctk.CTkSlider(
            bloc_duree, from_=5, to=60, number_of_steps=55,
            command=self._maj_etiquettes,
        )
        self.slider_duree.set(30)
        self.slider_duree.pack(fill="x", pady=(6, 0))
        self.val_duree = ctk.CTkLabel(
            bloc_duree, text="30 s", font=ctk.CTkFont(size=11),
            text_color=COULEUR_TEXTE_SECONDAIRE,
        )
        self.val_duree.pack(anchor="e")

        # --- bouton générer
        self.bouton_generer = ctk.CTkButton(
            bandeau, text="Générer la vidéo", height=46,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COULEUR_ACCENT, hover_color=COULEUR_ACCENT_SOMBRE,
            text_color="#04231a",
            command=self._lancer_generation,
        )
        self.bouton_generer.grid(row=0, column=3, padx=(6, 18), pady=14)

        # --- barre de progression
        self.progression = ctk.CTkProgressBar(
            bandeau, height=10, corner_radius=5,
            progress_color=COULEUR_ACCENT,
        )
        self.progression.set(0)
        self.progression.grid(row=1, column=0, columnspan=4, sticky="ew",
                              padx=18, pady=(0, 14))

        self.message = ctk.CTkLabel(
            bandeau, text="", anchor="w",
            font=ctk.CTkFont(size=12),
            text_color=COULEUR_TEXTE_SECONDAIRE,
        )
        self.message.grid(row=2, column=0, columnspan=4, sticky="ew",
                          padx=18, pady=(0, 12))

    # -------------------------------------------------------------- actions
    def _maj_etiquettes(self, _valeur: float = 0.0) -> None:
        self.val_expression.configure(text=f"{self.slider_expression.get():.2f}")
        self.val_lissage.configure(text=f"{int(self.slider_lissage.get())}")
        self.val_duree.configure(text=f"{int(self.slider_duree.get())} s")

    def _choisir_image(self) -> None:
        if self.en_cours:
            return
        from tkinter import filedialog
        chemin = filedialog.askopenfilename(
            title="Choisir l'image de référence",
            filetypes=FORMATS_IMAGE,
        )
        if not chemin:
            return
        self._charger_image(chemin)

    def _charger_image(self, chemin: str) -> None:
        try:
            image = cv2.imdecode(
                np.fromfile(chemin, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except Exception:
            image = None
        if image is None:
            self.apercu_image.afficher_message(
                "Image illisible", COULEUR_ERREUR)
            return

        # on limite les très grandes images (performance)
        h, l = image.shape[:2]
        if max(h, l) > 1920:
            echelle = 1920 / max(h, l)
            image = cv2.resize(image, (int(l * echelle), int(h * echelle)),
                               interpolation=cv2.INTER_AREA)

        points = self.detecteur.points_photo(image)
        if points is None:
            self.apercu_image.afficher_message(
                "Aucun visage détecté sur cette image",
                COULEUR_ERREUR)
            self.etat_image.configure(text=os.path.basename(chemin),
                                      text_color=COULEUR_ERREUR)
            return

        self.chemin_image = chemin
        self.image_reference = image
        self.points_reference = points
        self.apercu_image.afficher_image(image, os.path.basename(chemin))
        self.etat_image.configure(
            text=f"Visage détecté ({len(points)} points)",
            text_color=COULEUR_ACCENT,
        )
        self._maj_bouton()

    def _choisir_video(self) -> None:
        if self.en_cours:
            return
        from tkinter import filedialog
        chemin = filedialog.askopenfilename(
            title="Choisir la vidéo de mouvement",
            filetypes=FORMATS_VIDEO,
        )
        if not chemin:
            return
        self.chemin_video = chemin
        self.etat_video.configure(
            text=os.path.basename(chemin) + " — analyse en cours…",
            text_color=COULEUR_ALERTE,
        )
        self.apercu_video.afficher_message("Vidéo chargée, en attente d'analyse")
        self._maj_bouton()
        threading.Thread(target=self._analyser_video, daemon=True).start()

    def _analyser_video(self) -> None:
        """Analyse la vidéo dans un fil séparé (cache géré automatiquement)."""
        try:
            analyse = AnalyseVideo.charger_ou_analyser(
                self.chemin_video, self.detecteur,
                callback_progression=lambda i, t: self.apres(
                    lambda: self._progression_analyse(i, t)
                ),
            )
        except ValueError as exc:
            self.apres(lambda: self._echec_video(str(exc)))
            return
        except Exception:
            traceback.print_exc()
            self.apres(lambda: self._echec_video(
                "Erreur pendant l'analyse de la vidéo."))
            return

        self.analyse = analyse
        self.apres(lambda: self._succes_video())

    def _progression_analyse(self, i: int, total: int) -> None:
        if total > 0:
            self.progression.set(min(i / total, 1.0) * 0.4)
            self.message.configure(
                text=f"Analyse de la vidéo… {i}/{total} images")

    def _echec_video(self, raison: str) -> None:
        self.analyse = None
        self.apercu_video.afficher_message(raison, COULEUR_ERREUR)
        self.etat_video.configure(text=raison, text_color=COULEUR_ERREUR)
        self.message.configure(text="")
        self._maj_bouton()

    def _succes_video(self) -> None:
        duree = self.analyse.nb_frames / max(self.analyse.fps, 1.0)
        self.apercu_video.afficher_message(
            f"Vidéo analysée : {self.analyse.nb_frames} images, "
            f"{duree:.1f} s", COULEUR_ACCENT)
        self.etat_video.configure(
            text=f"OK — {self.analyse.nb_frames} images ({duree:.1f} s)",
            text_color=COULEUR_ACCENT,
        )
        self.message.configure(text="")
        self.progression.set(0)
        self._maj_bouton()

    def _maj_bouton(self) -> None:
        pret = (self.image_reference is not None
                and self.analyse is not None
                and not self.en_cours)
        self.bouton_generer.configure(
            state="normal" if pret else "disabled",
            text="Générer la vidéo" if not self.en_cours else "Génération…",
        )

    def _lancer_generation(self) -> None:
        if not (self.image_reference is not None
                and self.points_reference is not None
                and self.analyse is not None):
            return
        self.en_cours = True
        self._maj_bouton()
        self.badge_etat.configure(text="Traitement…", text_color=COULEUR_ALERTE)

        from tkinter import filedialog
        chemin = filedialog.asksaveasfilename(
            title="Enregistrer la vidéo animée",
            defaultextension=".mp4",
            filetypes=[("Vidéo MP4", "*.mp4")],
            initialfile="motion_control.mp4",
        )
        if not chemin:
            self.en_cours = False
            self.badge_etat.configure(text="Prêt", text_color=COULEUR_ACCENT)
            self._maj_bouton()
            return

        self.chemin_sortie = chemin
        amplification = self.slider_expression.get()
        lissage = int(self.slider_lissage.get())
        fps = max(self.analyse.fps, 1.0)
        limite = int(self.slider_duree.get() * fps)

        threading.Thread(
            target=self._generer,
            args=(chemin, amplification, lissage, limite),
            daemon=True,
        ).start()

    def _generer(self, chemin: str, amplification: float, lissage: int,
                 limite: int) -> None:
        try:
            generer_video(
                self.image_reference, self.points_reference, self.analyse,
                chemin_sortie=chemin,
                amplification=amplification,
                lissage=lissage,
                limite_frames=limite,
                callback=lambda i, t: self.apres(
                    lambda: self._progression_generation(i, t)
                ),
            )
        except Exception as exc:
            traceback.print_exc()
            self.apres(lambda: self._echec_generation(str(exc)))
            return
        self.apres(lambda: self._succes_generation(chemin))

    def _progression_generation(self, i: int, total: int) -> None:
        self.progression.set(0.4 + min(i / max(total, 1), 1.0) * 0.6)
        self.message.configure(
            text=f"Génération de la vidéo… {i}/{total} images")

    def _echec_generation(self, raison: str) -> None:
        self.en_cours = False
        self.badge_etat.configure(text="Échec", text_color=COULEUR_ERREUR)
        self.message.configure(text=raison, text_color=COULEUR_ERREUR)
        self.progression.set(0)
        self._maj_bouton()

    def _succes_generation(self, chemin: str) -> None:
        self.en_cours = False
        self.badge_etat.configure(text="Terminé", text_color=COULEUR_ACCENT)
        self.message.configure(
            text=f"Vidéo enregistrée : {chemin}", text_color=COULEUR_ACCENT)
        self.progression.set(1.0)
        self._maj_bouton()
        self._ouvrir_dossier(chemin)

    # ------------------------------------------------------------- utilitaires
    def apres(self, fonction) -> None:
        """Exécute une action sur le thread de l'interface."""
        self.after(0, fonction)

    @staticmethod
    def _ouvrir_dossier(chemin_fichier: str) -> None:
        dossier = os.path.dirname(os.path.abspath(chemin_fichier))
        try:
            if os.name == "nt":
                os.startfile(dossier)  # type: ignore[attr-defined]
            elif os.name == "posix":
                import subprocess
                subprocess.Popen(["xdg-open", dossier])
        except Exception:
            pass  # l'ouverture du dossier n'est jamais bloquante

    def _quitter(self) -> None:
        try:
            self.detecteur.fermer()
        except Exception:
            pass
        self.destroy()


def lancer_application() -> None:
    app = Application()
    app.mainloop()
