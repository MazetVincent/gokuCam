# Ajouter un signe

*English version: [../ADDING_A_SIGN.md](../ADDING_A_SIGN.md)*

Dans GokuCam, un signe et l'effet qu'il déclenche forment un **jutsu**. En ajouter un, c'est écrire une petite classe et l'enregistrer. L'application s'occupe du reste : détection du maintien et barre de progression, raccourci clavier, affichage de debug, image figée pendant que vous êtes invisible, et self-test.

- [1. Le contrat](#1-le-contrat)
- [2. Décrire le signe : `checks()`](#2-décrire-le-signe--checks)
- [3. Régler les seuils sur vos mains](#3-régler-les-seuils-sur-vos-mains)
- [4. Dessiner l'effet : `render()`](#4-dessiner-leffet--render)
- [5. Enregistrer le jutsu](#5-enregistrer-le-jutsu)
- [6. Tester](#6-tester)
- [7. Exemple complet](#7-exemple-complet)
- [Liste de vérification](#liste-de-vérification)

## 1. Le contrat

`gokucam/jutsus/base.py` :

```python
class Jutsu(ABC):
    name: str               # identifiant : "wave_goodbye"
    title: str              # indication dans l'aperçu : "Wave goodbye"
    short: str              # ligne des raccourcis : "Wave"
    key: str                # touche qui joue l'effet sans le signe : "w"
    hint: str               # comment faire le signe : "open palm, all fingers extended"
    vanish_seconds: float
    arrival_seconds: float

    def checks(self, signs: SignInput) -> dict[str, bool]: ...           # obligatoire
    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray: ...  # obligatoire
    def start(self, scene: Scene, arriving: bool) -> None: ...           # facultatif
    def hidden_image(self, scene: Scene) -> np.ndarray: ...              # facultatif, par défaut la pièce vide
    def leftover_box(self) -> Box | None: ...                            # facultatif
```

## 2. Décrire le signe : `checks()`

`checks()` renvoie des **critères nommés**. Le signe est fait quand ils sont tous `True`. Choisissez des noms lisibles : `--debug` affiche en direct ceux qui manquent (« Wave: missing pinky extended »).

### Ce que vous recevez

```python
@dataclass
class SignInput:
    hands: list[np.ndarray]       # repères (21, 2) de chaque main suivie, en pixels de l'image de détection 640×360
    face: Face | None             # face.forehead (x, y) et face.width ; None si aucun visage vu récemment
    still_hands: list[np.ndarray] # (paresseux) mains redétectées sur l'image fixe, voir plus bas
```

L'axe y va **vers le bas** : « au-dessus » veut dire un y plus petit.

### Repères de la main

```
    4 bout du pouce    8 bout index     12 bout majeur     16 bout annul.   20 bout auric.
    3                  7                11                 15               19
    2                  6 index PIP      10 majeur PIP      14 annul. PIP    18 auric. PIP
    1                  5 index MCP       9 majeur MCP      13 annul. MCP    17 auric. MCP
                                    0 poignet
```

Les constantes (`WRIST`, `INDEX_TIP`, `MIDDLE_MCP`…) sont dans `gokucam/hands.py`.

### Fonctions utiles (`gokucam/hands.py`)

| Fonction | Signification |
|---|---|
| `finger_extended(hand, "index")` | bout du doigt nettement plus loin du poignet que l'articulation du milieu |
| `finger_folded(hand, "ring")` | bout du doigt revenu au niveau de l'articulation du milieu |
| `index_up(hand)` | index tendu, pointe au moins une échelle de main au-dessus du poignet |
| `hand_scale(hand)` | distance poignet → articulation du majeur : **l'unité de toute distance sur une main** |
| `dist(hand, a, b)` | distance entre deux repères |
| `best_hand(hands, fn)` | applique `fn(hand)` à chaque main et renvoie les critères de celle qui en remplit le plus |
| `FINGERS` | `"index"`, `"middle"`, `"ring"`, `"pinky"` |

### Règles d'or

- **Exprimez chaque distance par rapport à un élément de l'image** : `hand_scale(hand)` pour les distances sur une main, `face.width` pour les distances au visage. Jamais de pixels bruts : le signe doit marcher à 40 cm comme à 1,5 m de la caméra.
- **Signes à une main** : enveloppez les critères dans `best_hand(signs.hands, lambda hand: {...})`.
- **Signes à deux mains** : regardez `seal_checks()` dans `jutsus/substitution.py`. Il choisit la paire de mains dont les pointes sont les plus proches.
- **Mains jointes ou superposées** : le suivi vidéo peut les fusionner en une seule main. Demandez `signs.still_hands` (une détection sur l'image fixe, sans suivi), mais **seulement quand la pose de la main fusionnée est vue** : cela coûte environ 10 ms par image.
- **Ne marchez pas sur les signes existants** : ajoutez un critère d'exclusion. Peace out a « away from the forehead » pour ne pas se déclencher quand vous faites la Transmission instantanée ; la Substitution a « below the eyes ».
- **Évitez les gestes du quotidien** : un signe que l'on fait sans y penser (se toucher le visage, un pouce levé en réunion) vous fera disparaître par accident. La Substitution exige deux mains précisément pour qu'un geste « chut » ne suffise pas.
- Le visage peut être `None` (les mains le cachent souvent). Décidez ce que cela signifie pour votre signe : la Transmission instantanée exige un visage ; la Substitution et Peace out traitent « pas de visage » comme « loin du visage ».

## 3. Régler les seuils sur vos mains

1. **En direct** : `./run.command --debug`. Chaque signe a une ligne dans l'aperçu avec ses critères manquants. Faites votre signe et regardez lequel reste manquant.
2. **Sur une photo** : prenez une photo du signe depuis votre place habituelle (une capture d'écran de l'aperçu GokuCam convient), puis :
   ```bash
   .venv/bin/python tools/inspect_sign.py photo.jpg --out annotee.jpg --json tests/data/mon_signe.json
   ```
   L'outil affiche chaque critère de chaque jutsu, enregistre la photo avec les repères dessinés (suivi vidéo en vert, détection sur image fixe en magenta) et sauve les repères comme jeu de données de test.
3. Regardez les valeurs qui comptent (les distances divisées par `hand_scale`) et placez le seuil **à mi-chemin** entre « signe fait » et la pose « signe pas fait » la plus proche, pour tolérer un signe approximatif sans attraper les gestes voisins.

C'est ainsi que le signe de la Substitution a été calibré : l'outil a montré que le suivi vidéo voyait **une** main fusionnée alors que l'image fixe en voyait deux, d'où `single_index_up` + `still_hands`.

## 4. Dessiner l'effet : `render()`

```python
def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray
```

Appelée à chaque image de l'effet, avec `t` qui va de 0 à 1 en `vanish_seconds` (départ) ou `arrival_seconds` (arrivée).

### Ce que vous recevez

```python
@dataclass
class Scene:
    frame: np.ndarray          # image de la webcam en direct (uint8 BGR, pleine résolution)
    plate: np.ndarray          # photo de la pièce vide
    hidden: np.ndarray         # ce que voit l'appel pendant l'invisibilité (photo de la pièce, ou un tronc…)
    mask: np.ndarray           # masque de la personne, float32 0..1, pleine résolution
    seed: int                  # fixe pendant tout l'effet : pour les dispositions aléatoires
    rng: np.random.Generator   # nouvelles valeurs à chaque image : pour le scintillement
    leftover: Box | None       # ce que le départ précédent a laissé dans `hidden`
    size                       # (largeur, hauteur)
    person_box                 # (x0, y0, x1, y1) boîte englobante du masque
```

### Les règles

- **Aucun saut aux deux bouts.** Départ : à `t = 0` l'image doit ressembler à `scene.frame`, et quand `t → 1` à `hidden_image(scene)` (par défaut `scene.plate`). Arrivée : à `t = 0` elle doit ressembler à `scene.hidden`, et quand `t → 1` à `scene.frame`. L'application passe à l'image figée (ou au direct) dès que `t` atteint 1.
- **À l'arrivée, partez de `scene.hidden`, pas de `scene.plate`** : on peut revenir après le départ d'un autre jutsu (par exemple avec un tronc à l'écran).
- **Restez sous environ 30 ms par image en 1280×720.** La détection prend déjà environ 20 ms. Faites les calculs lourds en résolution réduite (voir la fumée, calculée au 1/4) et le travail propre à l'effet une seule fois dans `start()` (voir le sprite du tronc).
- **Aléatoire stable ou vivant** : ce qui doit rester en place pendant l'effet vient de `np.random.default_rng(scene.seed)` ; le scintillement vient de `scene.rng`.
- **Ne toucher que la personne** : appliquez la même déformation à `scene.frame` et à `scene.mask`, puis composez sur la pièce (voir `render_teleport`). `gokucam.fx.remove_person(frame, plate, mask)` donne l'image sans la personne.

### Points d'extension facultatifs

- `start(scene, arriving)` : appelée une fois au début de l'effet. Figez la personne (Peace out garde un `Ghost`), générez des sprites, placez des particules.
- `hidden_image(scene)` : ce que voit l'appel une fois que vous avez disparu. La Substitution renvoie la pièce avec le tronc.
- `leftover_box()` : la zone où `hidden_image()` diffère de la photo de la pièce. Elle est transmise à l'arrivée suivante dans `scene.leftover` (la fumée de la Substitution couvre le tronc au retour).

Fonctions utiles dans `gokucam/fx.py` : `smoothstep(a, b, t)` pour minuter les phases, `remove_person`, `pixelate`, `fractal_noise`.

## 5. Enregistrer le jutsu

Dans `gokucam/jutsus/__init__.py` :

```python
from .wave_goodbye import WaveGoodbye

def default_jutsus() -> list[Jutsu]:
    return [InstantTransmission(), Substitution(), PeaceOut(), WaveGoodbye()]
```

Choisissez une `key` libre : `b`, `d`, `s` et `q` appartiennent à l'application, et `t`, `n`, `p` aux jutsus existants.

## 6. Tester

- **Le self-test joue votre effet automatiquement** (départ, arrivée, et croisements avec les autres jutsus), avec les vrais modèles sur des images synthétiques :
  ```bash
  ./run.command --self-test
  ```
- **Tests unitaires** du signe, avec les mains synthétiques de `tests/helpers.py`, ou le jeu de données enregistré avec `tools/inspect_sign.py` :
  ```python
  from gokucam.jutsus.wave_goodbye import WaveGoodbye
  from tests.helpers import FACE_ABOVE, make_hand, signs

  class WaveGoodbyeTest(unittest.TestCase):
      jutsu = WaveGoodbye()

      def test_open_palm(self):
          self.assertTrue(self.jutsu.detected(signs([make_hand(open_palm=True)], FACE_ABOVE)))

      def test_two_fingers_are_not_enough(self):
          self.assertFalse(self.jutsu.detected(signs([make_hand()], FACE_ABOVE)))
  ```
- **Tests de l'effet** : vérifiez la première et la dernière image (aucun saut), puis une ou deux phases clés. Voir `tests/test_effects.py`.
- `tests/test_app.py` montre comment faire tourner toute la machine à états avec un faux jutsu et de faux détecteurs, sans MediaPipe.

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

## 7. Exemple complet

`gokucam/jutsus/wave_goodbye.py` : une main ouverte, et vous vous estompez. Cet exemple a été vérifié de bout en bout.

```python
"""Wave goodbye: an open palm, and you fade away."""

from __future__ import annotations

import cv2
import numpy as np

from ..fx import smoothstep
from ..hands import FINGERS, best_hand, finger_extended
from .base import Jutsu, Scene, SignInput


class WaveGoodbye(Jutsu):
    name = "wave_goodbye"
    title = "Wave goodbye"
    short = "Wave"
    key = "w"
    hint = "open palm, all fingers extended"
    vanish_seconds = 0.8
    arrival_seconds = 0.8

    def checks(self, signs: SignInput) -> dict[str, bool]:
        return best_hand(signs.hands, lambda hand: {
            f"{finger} extended": finger_extended(hand, finger) for finger in FINGERS
        })

    def render(self, scene: Scene, t: float, arriving: bool) -> np.ndarray:
        absent = scene.hidden if arriving else scene.plate
        gone = smoothstep(0.0, 1.0, 1.0 - t if arriving else t)  # 0 = you, 1 = vanished
        room = cv2.addWeighted(scene.frame, 1.0 - gone, absent, gone, 0)
        you = (scene.mask * (1.0 - gone)).astype(np.float32)
        return cv2.blendLinear(scene.frame, room, you, 1.0 - you)
```

À `gone = 0`, le résultat est l'image en direct ; à `gone = 1`, c'est exactement `absent`. Les deux bouts se raccordent donc sans saut à l'image figée et au direct.

Attention : une main ouverte est un geste courant en réunion (saluer, expliquer). Pour un vrai signe, ajoutez un critère qui le rend volontaire, par exemple « main au-dessus de la tête » (`hand[WRIST][1] < face.forehead[1]`).

## Liste de vérification

- [ ] `checks()` n'utilise que des seuils relatifs (`hand_scale`, `face.width`)
- [ ] les noms des critères se lisent bien dans `--debug`
- [ ] le signe ne peut être confondu ni avec les signes existants ni avec un geste du quotidien
- [ ] `render()` commence et finit exactement sur les images attendues, au départ comme à l'arrivée
- [ ] moins d'environ 30 ms par image ; calculs lourds dans `start()` ou en résolution réduite
- [ ] enregistré dans `default_jutsus()` avec une `key` libre
- [ ] tests unitaires du signe ; `--self-test` passe
- [ ] les deux README mis à jour (tableau des signes)
