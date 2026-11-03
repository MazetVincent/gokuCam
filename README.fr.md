# GokuCam

**Une webcam virtuelle pour Teams, Slack, Zoom ou Meet : faites un signe de la main et disparaissez de l'appel.**

GokuCam se place entre votre vraie webcam et votre application de visio. Elle observe vos mains. Quand vous faites l'un des signes ci-dessous, elle joue un effet de disparition façon anime, puis montre à l'appel une photo figée de votre pièce vide, jusqu'à ce que vous refassiez un signe.

*Read in English: [README.md](README.md)*

| Signe | Effet |
|---|---|
| **Transmission instantanée** (Goku) : index et majeur collés sur le front | Des lignes d'énergie bleues apparaissent autour de vous, des bandes de votre corps se déchirent verticalement, puis tout votre corps se resserre en une fine colonne et disparaît |
| **Substitution** (Naruto) : mains jointes doigts croisés, les deux index tendus et collés vers le haut, devant le menton | Vous disparaissez dans un nuage de fumée ; quand il se dissipe, un tronc d'arbre se tient debout là où vous étiez assis |
| **Peace out** : un V, loin du visage | Vous devenez un fantôme translucide en blocs de compression, quelques blocs bavent sur le côté, et vous n'êtes plus là (le mème « bye ») |
| **Dans le buisson** (Homer Simpson) : un doigt levé au-dessus de la tête | Une haie cartoon pousse autour de vous et au-dessus, puis se referme sur vous, le visage en dernier ; le retour la joue à l'envers |

Refaites n'importe quel signe pour revenir : le retour joue l'effet d'arrivée de ce signe.

Et des effets qui ne font pas disparaître : des **overlays**, qui durent tant que leur signe est tenu :

| Signe | Effet |
|---|---|
| **This is fine** : prenez votre tasse en main | Des flammes cartoon apparaissent l'une après l'autre à différents endroits autour de vous (toujours derrière vous), un plafond de fumée envahit le haut de l'écran, et « This is fine. » s'écrit lettre par lettre |
| **Absolute cinema** : les deux paumes ouvertes levées au-dessus de la tête | L'image passe en noir et blanc, des rideaux de théâtre rouges se ferment derrière vous et « ABSOLUTE CINEMA » apparaît ; baissez les mains et tout se rejoue à l'envers |

Tout tourne en local. Aucune image n'est enregistrée ni envoyée ailleurs qu'à l'appel lui-même. Le seul accès réseau est le téléchargement, au premier lancement, de quatre modèles MediaPipe officiels de Google.

## Sommaire

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Fonctionnement (aperçu)](#fonctionnement-aperçu)
- [Organisation du projet](#organisation-du-projet)
- [Ajouter votre propre signe](#ajouter-votre-propre-signe)
- [Tests](#tests)
- [Dépannage](#dépannage)
- [Crédits](#crédits)

## Prérequis

- **macOS sur Apple Silicon** (développé et testé sur un MacBook M1). Le code vise aussi Windows et Linux (MediaPipe et pyvirtualcam les prennent en charge), mais n'y a pas été testé.
- **Python 3.10 à 3.14**.
- **[OBS Studio](https://obsproject.com/) 30+**, uniquement pour son pilote de caméra virtuelle. Sous Linux, [v4l2loopback](https://github.com/umlaeute/v4l2loopback) fonctionne à la place.
- Une webcam.

## Installation

1. **Installez OBS Studio** et activez sa caméra virtuelle une fois :
   ```bash
   brew install --cask obs
   ```
   Ouvrez OBS et cliquez sur **Démarrer la caméra virtuelle**. macOS demande d'autoriser une extension système dans *Réglages Système > Confidentialité et sécurité*. Cliquez ensuite sur **Arrêter la caméra virtuelle** et quittez OBS. Vous n'en aurez plus besoin : c'est GokuCam qui pilote la caméra virtuelle.

2. **Récupérez le code** et lancez-le :
   ```bash
   git clone https://github.com/MazetVincent/gokuCam.git
   cd gokuCam
   ./run.command
   ```
   Le premier lancement crée un environnement virtuel dans `.venv/`, installe les dépendances et télécharge les modèles dans `models/` (environ 13 Mo). Sur macOS, vous pouvez aussi double-cliquer sur `run.command` dans le Finder.

3. **Autorisez l'accès à la caméra** quand macOS le demande. L'autorisation va à l'application qui lance GokuCam : Terminal, iTerm, votre IDE…

## Utilisation

1. **Sortez du champ** pendant le compte à rebours de 5 secondes : GokuCam photographie votre pièce vide. La photo s'affiche ensuite en vignette pendant 4 secondes ; si vous y apparaissez, appuyez sur `B` pour la refaire.
2. Dans votre application de visio, choisissez la caméra **OBS Virtual Camera** :
   - **Teams** : *Paramètres > Appareils > Caméra*
   - **Slack** (huddles) : la flèche à côté du bouton caméra, ou *Préférences > Audio et vidéo*
   - Quittez et rouvrez l'application (Cmd+Q) si la caméra n'apparaît pas.
3. **Tenez un signe environ 0,3 s** pour disparaître, puis de nouveau pour revenir. Prenez votre tasse pour « This is fine » ; reposez-la et le feu s'éteint.

Gardez la fenêtre d'aperçu ouverte pendant l'appel : la fermer coupe la caméra virtuelle. Les indications et le debug n'apparaissent que dans l'aperçu ; l'appel reçoit toujours une image propre.

| Touche (dans l'aperçu) | Action |
|---|---|
| `T` / `N` / `P` / `H` | Jouer Transmission instantanée / Substitution / Peace out / Dans le buisson sans faire le signe |
| `F` / `C` | Allumer ou éteindre « This is fine » / « Absolute cinema » sans faire le signe |
| `B` | Refaire la photo de la pièce |
| `D` | Debug : pour chaque signe, affiche les critères qui manquent |
| `S` | Enregistrer une capture de diagnostic (caméra, sortie, photo de la pièce, masque) dans `captures/` |
| `Q` / `Échap` | Quitter |

Options en ligne de commande (`./run.command --help`) :

| Option | Rôle |
|---|---|
| `--debug` | Démarrer avec l'affichage de debug |
| `--camera N` | Forcer l'index de la webcam (par défaut la première vraie caméra ; les caméras virtuelles sont ignorées) |
| `--width`, `--height`, `--fps` | Format de capture et de sortie (par défaut 1280×720 à 30 i/s) |
| `--no-virtual` | Aperçu seul, sans caméra virtuelle |
| `--backend` | Backend pyvirtualcam : `obs`, `unitycapture` ou `v4l2loopback` |
| `--no-mirror` | Ne pas inverser l'aperçu |
| `--self-test` | Charger les modèles et jouer tous les effets sur des images synthétiques, sans webcam |

**Conseils**
- La photo de la pièce est figée. Si la lumière change (lampe allumée, nuit qui tombe), votre retour le montrera : appuyez sur `B` pour refaire la photo.
- Ne déplacez ni la caméra ni votre fauteuil après la photo.
- Une lumière venant de derrière l'écran (face à vous) donne les meilleurs résultats.

## Fonctionnement (aperçu)

```
webcam ──> détection (640×360) ──> signes ──> machine à états ──> effet ──> caméra virtuelle ──> Teams / Slack
            · masque de la personne             photo → visible → disparition → caché → retour → visible
            · repères des mains                                                + overlays par-dessus
            · visage
            · objets (tasse)
```

- **Détection** : des modèles MediaPipe tournent sur une copie réduite de chaque image : un segmenteur de personne (le masque), un détecteur de repères de la main (21 points par main), un détecteur de visage (pour situer le front) et, quand une main est visible, un détecteur d'objets (pour repérer une tasse) sur un fil d'exécution en arrière-plan.
- **Signes** : chaque signe est un ensemble de critères géométriques nommés sur les repères, par exemple « index tendu » ou « doigts collés ». Les seuils sont relatifs à la taille de la main, donc les signes marchent près comme loin de la caméra. Un signe doit être tenu 0,35 s et ne se déclenche qu'une fois, jusqu'à ce qu'on le relâche.
- **Invisibilité** : plutôt que de vous effacer de l'image en direct, ce qui laisse ombres, halos et variations d'exposition, GokuCam envoie à l'appel **une photo de la pièce vide** prise au démarrage, avec un léger grain de capteur animé. La vidéo en direct ne revient qu'au signe suivant.
- **Effets** : chaque effet est une fonction de `t`, de 0 à 1, qui compose l'image en direct, le masque de la personne et la photo de la pièce avec OpenCV et NumPy. Ils tournent en temps réel : 7 à 28 ms par image sur un M1, en plus d'environ 30 ms de détection.
- **Overlays** : ils vivent en dehors du cycle disparition / retour. Ils apparaissent en fondu quand leur signe est tenu, durent tant qu'il l'est (et un instant de plus, pour que cacher la tasse une seconde n'éteigne pas le feu), et se dessinent par-dessus ce que reçoit l'appel.

Toute l'histoire, y compris les choix de conception et ce qui n'a pas marché, est dans **[docs/fr/FONCTIONNEMENT.md](docs/fr/FONCTIONNEMENT.md)**.

## Organisation du projet

```
gokucam/
  cli.py            ligne de commande, boucle principale, self-test
  app.py            machine à états : photo de la pièce, signes, effets, image figée
  models.py         téléchargement des modèles MediaPipe et détecteurs
  hands.py          indices des repères de la main, géométrie des doigts, suivi du geste
  camera.py         choix de la vraie webcam, ouverture de la caméra virtuelle
  preview.py        incrustations de la fenêtre d'aperçu, captures de diagnostic
  fx.py             petits utilitaires d'image (smoothstep, bruit, pixellisation…)
  jutsus/
    base.py         le contrat d'un jutsu : checks() + render()
    instant_transmission.py
    substitution.py
    peace_out.py
    into_the_bush.py
    __init__.py     registre des jutsus
  overlays/
    base.py         le contrat d'un overlay : checks() + render(level)
    this_is_fine.py
    absolute_cinema.py
    __init__.py     registre des overlays
tests/              tests unitaires (sans webcam ni MediaPipe)
tools/              inspect_sign.py : ce que les détecteurs voient d'un signe sur une photo
docs/               fonctionnement, ajout d'un signe (anglais et français)
run.command         lanceur (crée .venv au premier lancement)
```

## Ajouter votre propre signe

Un signe et son effet forment un **jutsu** : une petite classe avec deux méthodes.

```python
class Wave(Jutsu):
    name, title, short, key, hint = "wave", "Wave goodbye", "Wave", "w", "open palm, fingers spread"
    vanish_seconds = arrival_seconds = 0.8

    def checks(self, signs):                 # critères nommés ; le signe est fait quand tous sont vrais
        return best_hand(signs.hands, lambda hand: {
            "all fingers extended": all(finger_extended(hand, f) for f in FINGERS),
        })

    def render(self, scene, t, arriving):    # image t de l'effet, t de 0 à 1
        ...
```

Ajoutez-le à `default_jutsus()` dans `gokucam/jutsus/__init__.py`, et l'application fait le reste : maintien du signe, raccourci clavier, debug, image figée, self-test. Un overlay fonctionne de la même façon (`gokucam/overlays/`), avec un `render()` qui dessine par-dessus l'image à un niveau de fondu donné. Le guide pas à pas (carte des repères, fonctions de géométrie, objets comme la tasse, réglage des seuils sur vos mains, tests) est dans **[docs/fr/AJOUTER_UN_SIGNE.md](docs/fr/AJOUTER_UN_SIGNE.md)**.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .   # 68 tests, quelques secondes, sans webcam
./run.command --self-test                             # tout le pipeline avec les vrais modèles
```

Les tests couvrent les signes (y compris des repères enregistrés sur une vraie photo du signe de la Substitution), la première et la dernière image de chaque effet ainsi que ses phases clés, l'overlay (flammes derrière vous, sous-titre tapé, fondus), la machine à états avec un faux jutsu et un faux overlay, la photo de la pièce et le choix de la caméra.

## Dépannage

| Symptôme | Solution |
|---|---|
| « No usable webcam » | Autorisez la caméra pour l'application qui lance GokuCam (*Réglages Système > Confidentialité et sécurité > Caméra*), puis relancez |
| « No virtual camera: preview only » | Activez une fois la caméra virtuelle d'OBS (voir Installation), puis quittez OBS |
| La caméra n'apparaît pas dans Teams / Slack | Quittez complètement l'application (Cmd+Q) et rouvrez-la pendant que GokuCam tourne |
| Un bref flash magenta dans l'appel au démarrage | Normal : GokuCam vérifie qu'il ne lit pas sa propre sortie |
| Un signe ne se déclenche pas | Appuyez sur `D` et lisez le critère qui manque ; voir [docs/fr/AJOUTER_UN_SIGNE.md](docs/fr/AJOUTER_UN_SIGNE.md#3-régler-les-seuils-sur-vos-mains) |
| Vous apparaissez sur la photo de la pièce | Appuyez sur `B` et sortez complètement du champ pendant le compte à rebours |
| « This is fine » ne démarre pas | Appuyez sur `D` : « mug in view » veut dire que le détecteur ne reconnaît pas votre tasse (changez d'angle, gardez-la dans la lumière) ; « hand on the mug » veut dire que votre main n'est pas vue dessus |

## Crédits

- [MediaPipe](https://ai.google.dev/edge/mediapipe) (Google) pour la segmentation, les repères de la main, la détection du visage et des objets
- [OpenCV](https://opencv.org/) et [NumPy](https://numpy.org/) pour tout le traitement d'image, [Pillow](https://python-pillow.org/) pour le texte du sous-titre
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) et [OBS Studio](https://obsproject.com/) pour la caméra virtuelle

Les effets s'inspirent de *Dragon Ball Z*, de *Naruto*, des *Simpson*, du mème « bye », de la BD « This is fine » de K.C. Green et du mème « Absolute cinema ». C'est un projet de fan, sans lien avec leurs auteurs ou éditeurs ; aucune image de ces œuvres n'est incluse : tout est dessiné en code.

## Licence

[MIT](LICENSE)
