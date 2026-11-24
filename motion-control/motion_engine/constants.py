"""Constantes partagées du moteur Motion Control."""

# Indice des landmarks MediaPipe FaceMesh
INDICE_NEZ = 1            # bout du nez
INDICE_MENTON = 199       # bas du menton
INDICE_FRONT = 10         # haut du front
INDICE_OEIL_GAUCHE = 33   # coin externe oeil gauche
INDICE_OEIL_DROIT = 263   # coin externe oeil droit

# Leviers d'animation (valeurs par défaut, modifiables dans l'interface)
INTENSITE_EXPRESSION = 1.2
AMPLITUDE_ROTATION = 1.0
LISSEMENT = 0.35          # 0 = brut, plus haut = plus lisse

# Palette de l'interface (identité sobre, verts profonds)
COULEUR_FOND = "#0d1512"
COULEUR_PANNEAU = "#16211c"
COULEUR_ACCENT = "#2dd4a7"
COULEUR_ACCENT_SOMBRE = "#0f9b7a"
COULEUR_ALERTE = "#e0a63a"
COULEUR_ERREUR = "#e05a5a"
COULEUR_TEXTE = "#e8f0ec"
COULEUR_TEXTE_SECONDAIRE = "#8fa69c"

FORMATS_IMAGE = [("Images", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"))]
FORMATS_VIDEO = [("Vidéos", ("*.mp4", "*.avi", "*.mov", "*.mkv"))]
