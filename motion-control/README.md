# Motion Control

**Anime une image de référence avec les mouvements d'une vidéo.**

Vous choisissez une photo (avec un visage), puis une vidéo où une personne parle,
sourit, hoche la tête… Motion Control détecte les mouvements du visage dans la
vidéo et les applique à votre photo, puis génère un fichier vidéo MP4 animé.

Fonctionne 100 % en local, aucune donnée n'est envoyée sur Internet.

---

## Obtenir le programme Windows (fichier .exe)

Ce dépôt contient un flux de construction automatique (GitHub Actions) qui
compile l'exécutable Windows 64 bits à chaque mise à jour du code.

1. Importez ce projet sur GitHub (bouton **Code → Upload files**, ou `git push`).
2. Ouvrez l'onglet **Actions** du dépôt.
3. Attendez la fin de la tâche **Build Windows EXE** (comptez quelques minutes).
4. Dans la page de la tâche, section **Artifacts**, téléchargez
   **MotionControl-win64** : vous y trouverez `MotionControl.exe`.

Pour créer une **Release téléchargeable publiquement** (lien permanent), créez
un tag de version :

```bash
git tag v1.0.0
git push origin v1.0.0
```

Le fichier `.exe` est alors attaché automatiquement à la release GitHub.

---

## Utilisation du programme

1. Lancez `MotionControl.exe` (premier lancement : quelques secondes de chargement).
2. **Image de référence** — cliquez sur « Choisir une image… » et sélectionnez
   une photo nette, de face, avec le visage bien visible.
3. **Vidéo de mouvement** — cliquez sur « Choisir une vidéo… ». La vidéo est
   analysée automatiquement (le résultat est mis en cache : une seconde analyse
   de la même vidéo est instantanée).
4. Réglez si besoin :
   - **Intensité de l'expression** : 1.0 = fidèle à la vidéo, 1.5 = très expressif.
   - **Lissage du mouvement** : 0 = brut, 6 = très doux (utile pour les vidéos tremblantes).
   - **Durée maximale** : limite la longueur de la vidéo générée.
5. Cliquez sur **Générer la vidéo**, choisissez où enregistrer le fichier.
6. À la fin, le dossier de sortie s'ouvre automatiquement avec votre `.mp4`.

### Conseils pour de meilleurs résultats

- Photo de référence : visage de face, bien éclairé, net, bouche fermée de préférence.
- Vidéo source : visage visible en permanence, pas de coupures, cadrage stable.
- Formats acceptés : images PNG / JPG / JPEG / BMP / WEBP ; vidéos MP4 / AVI / MOV / MKV.
- Les vidéos sont limitées à 60 secondes et à la résolution de l'image de référence.

---

## Compilation depuis les sources

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py                  # lancer l'application
```

Pour compiler vous-même le `.exe` :

```bash
pip install pyinstaller
pyinstaller --clean --noconfirm MotionControl.spec
# Résultat : dist/MotionControl.exe
```

---

## Structure du projet

```
motion-control/
├── main.py                        Point d'entrée
├── MotionControl.spec             Configuration PyInstaller
├── requirements.txt               Dépendances
├── motion_engine/                 Moteur d'animation
│   ├── face_analysis.py           Détection des visages (MediaPipe FaceMesh)
│   ├── video_analysis.py          Analyse frame par frame de la vidéo + cache
│   ├── motion_transfer.py         Transfert de mouvement (warp triangulé)
│   └── constants.py               Réglages et couleurs
├── interface/                     Interface graphique (CustomTkinter)
│   ├── main_window.py             Fenêtre principale
│   └── preview_panel.py           Panneaux d'aperçu
└── .github/workflows/build.yml    Construction automatique du .exe
```

## Comment ça marche

1. **Détection** — MediaPipe FaceMesh repère 468 points caractéristiques du
   visage (contours bouche, sourcils, yeux, nez…) sur la photo de référence et
   sur chaque frame de la vidéo.
2. **Alignement** — chaque pose vidéo est recalée sur la pose de référence
   (échelle + rotation + translation, décomposition de Procruste) : le fond,
   les cheveux et les épaules de la photo ne bougent pas.
3. **Amplification** — les mouvements fins (ouverture de bouche, sourcils,
   clignements) sont extraits de la vidéo et amplifiés selon le réglage choisi.
4. **Déformation** — le visage est découpé en triangles de Delaunay ; chaque
   triangle subit la transformation affine qui l'amène de la pose de référence
   vers la pose cible.
5. **Fusion** — le visage animé est fondu dans l'image d'origine via un masque
   flouté, pour un résultat sans couture.

## Prérequis

- Windows 10 / 11 (64 bits) pour l'exécutable.
- Python 3.10 ou 3.11 pour exécuter depuis les sources.
- Une webcam n'est pas nécessaire ; tout se fait à partir de fichiers.

## Licence

MIT — voir le fichier [LICENSE](LICENSE).
