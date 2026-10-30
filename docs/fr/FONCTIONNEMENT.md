# Comment fonctionne GokuCam

*English version: [../HOW_IT_WORKS.md](../HOW_IT_WORKS.md)*

Ce document suit une image de la webcam à travers le programme, puis décrit chaque effet et les choix de conception qui les sous-tendent, y compris les approches essayées puis abandonnées.

- [1. La boucle d'images](#1-la-boucle-dimages)
- [2. La détection](#2-la-détection)
- [3. Reconnaître les signes](#3-reconnaître-les-signes)
- [4. La photo de la pièce et l'invisibilité](#4-la-photo-de-la-pièce-et-linvisibilité)
- [5. La machine à états](#5-la-machine-à-états)
- [6. Les effets](#6-les-effets)
- [7. Webcam et caméra virtuelle](#7-webcam-et-caméra-virtuelle)
- [8. Performances](#8-performances)
- [9. Vie privée](#9-vie-privée)

## 1. La boucle d'images

`cli.run()` lit la webcam en 1280×720 à 30 i/s et, pour chaque image :

1. `GokuCam.step(frame)` (dans `app.py`) lance la détection, met à jour chaque signe, fait avancer la machine à états et renvoie **l'image destinée à l'appel**.
2. Cette image part vers la caméra virtuelle (`pyvirtualcam` → OBS Virtual Camera → Teams / Slack).
3. `preview.build_preview()` dessine, dans la fenêtre d'aperçu locale, une copie inversée avec les indications, la barre de maintien du signe et le debug éventuel. **Ces incrustations n'atteignent jamais l'appel.**
4. Les raccourcis clavier sont traités (`T`/`N`/`P`, `B`, `D`, `S`, `Q`).

## 2. La détection

`models.Detectors` regroupe trois modèles MediaPipe officiels, téléchargés une fois dans `models/` :

| Modèle | Sortie | Sert à |
|---|---|---|
| `selfie_segmenter_landscape` | probabilité « personne » par pixel | le masque de la personne : là où s'appliquent les effets |
| `hand_landmarker` | 21 points par main, 2 mains max | les signes |
| `blaze_face_short_range` | boîte et 6 points clés du visage | situer le front |

Ils tournent en **mode vidéo** sur une copie **640×360** de l'image : le suivi rend les repères stables et peu coûteux d'une image à l'autre, et la petite taille garde la détection autour de 20 ms par image sur un M1.

**Masque de la personne** (`app.refine_mask`) : la segmentation brute est seuillée généreusement, dilatée de 15 px, floutée, et ne peut pas s'effacer trop vite (`max(masque, 0,75 × masque précédent)`). Mieux vaut que les effets touchent un peu de décor que rater un bout de vous.

**Front** (`hands.face_from_detection`) : BlazeFace donne les yeux et la bouche. Le front est extrapolé au-dessus des yeux, dans l'axe bouche → yeux, à 0,8 fois l'écart entre les yeux ; il suit donc l'inclinaison de la tête. Les mains cachent souvent le visage : le dernier visage vu est conservé une seconde.

**Détection des mains sur l'image fixe** : le suivi vidéo a un angle mort. Quand les mains sont jointes doigts croisés (le signe de la Substitution), il suit **une seule main fusionnée** et ne cherche jamais la seconde, quel que soit le seuil de confiance. Un second détecteur de mains en **mode image** (sans suivi, confiance 0,3) trouve bien les deux mains sur la même image. Il ne tourne qu'à la demande (voir [3.3](#33-un-signe-à-deux-mains--la-substitution)), car il coûte environ 10 ms.

## 3. Reconnaître les signes

### 3.1 Des critères sur les repères

Un signe est un dictionnaire de **critères booléens nommés**, calculés à partir des repères (`Jutsu.checks()`). Le signe est fait quand ils sont tous vrais. Leurs noms sont ce qu'affiche `--debug` : « Naruto: missing both index up, index tips touching ».

Les briques de base sont dans `hands.py` :

- `hand_scale(hand)` : distance poignet → articulation du majeur. Elle varie peu avec la pose, donc **chaque seuil de distance en est un multiple**, et un signe marche que votre main soit à 40 cm ou à 1,5 m de la caméra.
- `finger_extended(hand, finger)` : le bout du doigt est nettement plus loin du poignet que son articulation du milieu (×1,1).
- `finger_folded(hand, finger)` : le bout du doigt revient au niveau de l'articulation du milieu (×1,05).
- `index_up(hand)` : index tendu et pointe au moins une échelle de main au-dessus du poignet.
- Les distances au visage sont des multiples de la largeur du visage.

Par exemple, la Transmission instantanée, c'est : index et majeur tendus, annulaire et auriculaire pliés, pointes à moins de 0,5 échelle de main l'une de l'autre, et à moins de 0,55 largeur de visage du front. Peace out reprend la même main, mais avec les pointes écartées de **plus** de 0,55 échelle de main, et à plus de 0,8 largeur de visage du front. Les deux signes ne peuvent pas être confondus.

### 3.2 Tenir un signe

`hands.GestureTracker` transforme « le signe est visible sur cette image » en « déclenche maintenant » :

- le signe doit être tenu **0,35 s** (la barre de progression de l'aperçu) ;
- les trous de détection jusqu'à 0,15 s ne remettent pas le compteur à zéro (les repères scintillent) ;
- après un déclenchement, le signe doit être **relâché 0,5 s** avant de pouvoir se redéclencher. Le tenir pendant tout l'effet ne vous fait donc pas revenir aussitôt.

### 3.3 Un signe à deux mains : la Substitution

Le signe de la Substitution a été calibré sur une vraie photo (ses repères sont enregistrés dans `tests/data/joined_hands.json`) :

1. Le suivi vidéo voit une main, index levé, autres doigts pliés (`single_index_up`).
2. Dans ce cas seulement, `SignInput.still_hands` lance la détection sur l'image fixe, qui trouve les deux mains.
3. Les critères : deux mains, les deux index levés, majeur et annulaire pliés sur au moins une main (la main de derrière est en partie cachée, ses doigts prédits ne sont pas fiables), pointes des index à moins de 0,6 échelle de main, et sous les yeux.

Un geste « chut » (un doigt sur la bouche) ressemble à l'étape 1, mais la détection sur image fixe ne trouve qu'une main : il ne déclenche jamais rien.

## 4. La photo de la pièce et l'invisibilité

### 4.1 Ce que voit l'appel pendant que vous êtes invisible

L'appel reçoit **une image figée** : la photo de la pièce vide, ou ce que le départ a laissé (le tronc, pour la Substitution). Six variantes avec un léger grain gaussien de capteur (σ = 1,5) défilent, pour que l'image ressemble à un flux en direct plutôt qu'à une photo. La webcam en direct ne sert plus qu'à guetter le signe suivant.

C'est le choix de conception le plus important du projet, et il est né d'un échec. L'effacement de la personne dans l'image en direct (remplacement du fond en temps réel) a longtemps été essayé :

1. **Fond appris pixel par pixel** : le fond n'est moyenné que là où personne ne se tient. Les zones jamais vues (derrière un fauteuil gaming où l'on reste assis toute la journée) devaient être inventées par inpainting, ce qui laissait une tache floue couleur peau.
2. **Ombres portées** : le segmenteur détecte les personnes, pas leurs ombres. Une lampe projetait une ombre sur la porte qui restait visible une fois « invisible », et les ombres mobiles se moyennaient dans le fond appris. Un détecteur d'ombre a été ajouté (plus sombre que le fond, même teinte, mesuré sur de vraies captures : rapport de luminosité ≈ 0,66, écart de teinte ≈ 0,03).
3. **Exposition automatique** : quand on s'approche en t-shirt clair, la webcam assombrit toute l'image. Le fond appris devenait un patchwork, et le détecteur d'ombre se déclenchait sur la moitié de l'image.

La photo figée règle les trois problèmes d'un coup : pas de masque en direct, pas d'ombres, pas de dérive d'exposition. En prime, le fauteuil vide rend la disparition plus convaincante. Le prix à payer : un gros changement de lumière pendant que vous êtes invisible se verra à votre retour ; `B` refait la photo.

### 4.2 Prendre la photo

`GokuCam.capture_plate` : un compte à rebours de 5 secondes, puis la **médiane des 15 dernières images**. La médiane efface le bruit du capteur et une personne qui traverse brièvement le champ. La photo s'affiche ensuite en vignette pendant 4 secondes pour que vous la vérifiiez.

Une version précédente attendait que les détecteurs voient une pièce vide. En pratique, un tableau de canards et un fauteuil gaming étaient détectés comme des personnes, et la photo n'était jamais prise. Un compte à rebours avec vérification visuelle est prévisible ; on ne fait confiance aux détecteurs que pour ce qu'ils font bien.

## 5. La machine à états

```
plate ──(compte à rebours, photo)──> visible
visible ──signe──> vanishing ──(effet terminé)──> hidden
hidden  ──signe──> appearing ──(effet terminé)──> visible
```

- `trigger(name)` choisit le jutsu, tire une **graine** (les dispositions aléatoires, comme les lignes d'énergie, restent identiques pendant tout l'effet, tandis que le générateur `rng` de chaque image fait scintiller) et appelle `jutsu.start(scene, arriving)`.
- Pendant un effet, `jutsu.render(scene, t, arriving)` est appelé avec `t = temps écoulé / durée`.
- À la fin d'un départ, `jutsu.hidden_image(scene)` devient l'image figée, et `jutsu.leftover_box()` note ce qu'il a laissé (le tronc). L'arrivée suivante le reçoit dans `scene.leftover`, pour que la fumée de la Substitution couvre aussi un tronc laissé par un départ précédent.
- On peut partir avec un jutsu et revenir avec un autre : chaque arrivée part de `scene.hidden`, quoi qu'il l'ait produit.

Une `Scene` contient tout ce dont un effet a besoin : l'image en direct, la photo de la pièce, l'image cachée actuelle, le masque de la personne en pleine résolution (et sa boîte englobante), la graine, le générateur aléatoire de l'image et la boîte du reste laissé.

## 6. Les effets

Tous les effets sont des fonctions pures de leurs entrées et de `t`, écrites avec OpenCV et NumPy. Chacun est testé sur sa première et sa dernière image (le départ doit commencer sur l'image en direct et finir **exactement** sur l'image cachée, pour éviter tout saut) et sur ses phases clés.

### 6.1 Transmission instantanée (`jutsus/instant_transmission.py`)

Le minutage vient de l'analyse image par image d'extraits de référence :

| t (départ) | Phase |
|---|---|
| 0 → 0,45 | **Lignes d'énergie** : 16 lignes verticales autour de la personne, cœur blanc et halo bleu électrique, qui scintillent puis se fragmentent en tirets et en points |
| 0,3 → 0,55 | **Déchirure partielle** : 30 % des bandes verticales du corps sont étirées et décalées ; certaines deviennent des trous qui laissent voir la pièce à travers vous |
| 0,55 → 1 | **Déchirure totale** : toutes les bandes partent, le corps **se resserre sur son axe vertical** en une fine colonne et disparaît |

Techniques :
- **Lanières** (`strand_field`) : l'image est découpée en bandes de 4 à 12 px. Chaque bande déchirée est étirée autour d'un point d'ancrage aléatoire (`y' = ancre + (y − ancre) × étirement`) et décalée, via une carte y pour `cv2.remap`. Étirer plutôt que flouter transforme les détails sombres (cheveux, barbe) en traînées nettes, comme dans l'anime. La carte est purement verticale : chaque colonne reste à sa place.
- **Seule la personne se déchire** : l'image en direct **et le masque de la personne** subissent le même remap, puis le résultat est composé sur la pièce. La pièce elle-même n'est jamais déformée.
- **Resserrement** : une carte x `x' = cx + (x − cx) / resserrement` ramène le corps sur son axe.
- **Lignes d'énergie** : ajouter de la lumière sature au blanc sur un mur clair. Les lignes **teintent** donc vers le bleu (`sortie × (1 − halo) + bleu × halo`), puis un cœur blanc est dessiné par-dessus. Elles restent bleues sur n'importe quel fond.
- **L'arrivée n'est pas un rembobinage** : elle se joue dans l'ordre chronologique (d'abord les lignes, puis le corps se déploie depuis une colonne, puis quelques bandes se recollent). Un rembobinage mettrait les lignes à la fin.

### 6.2 Substitution (`jutsus/substitution.py`)

| t (départ, 1,8 s) | Phase |
|---|---|
| 0 → 0,12 | La fumée éclate sur la personne et autour |
| 0,08 → 0,2 | Sous la fumée, l'image en direct laisse place à la photo de la pièce |
| 0,15 → 0,45 | Le tronc apparaît sous la fumée, un peu plus haut et penché, puis se pose |
| 0,3 → 0,9 | La fumée monte, s'effiloche en volutes et se dissipe, révélant le tronc |

**Fumée volumétrique** (`render_smoke`), calculée sur une grille au 1/4 de la résolution :
- 44 bouffées gaussiennes réparties sur la personne, plus denses au centre. Elles grossissent, montent et s'écartent, et chacune s'évanouit à son propre moment.
- La densité est modulée par un **bruit « chou-fleur »** (`1 − |2 × bruit fractal − 1|`), qui donne des volutes arrondies plutôt qu'un brouillard uniforme. Le bruit défile vers le haut au fil du temps.
- **L'érosion** retire d'abord les zones fines : la fumée s'effiloche au lieu de pâlir uniformément.
- **Éclairage** : un point est d'autant plus sombre qu'il y a de la fumée juste au-dessus (densité floutée décalée vers le bas), et les bosses de la turbulence accrochent la lumière. La couleur va des creux gris-bleu (168, 171, 177) aux sommets blancs (246, 246, 245).
- Opacité : `alpha = 1 − exp(−3 × densité)`.

**Tronc procédural** (`make_log_sprite`), généré une fois par départ :
- Cylindre vertical d'environ la moitié de la largeur des épaules, debout en bas au centre là où était la personne, le haut à hauteur du torse, le bas coupé par le bord de l'image.
- **Écorce** : un bruit fractal étiré 10× verticalement forme des plaques verticales, séparées par des fissures sombres et douces ; grain fin ; ombrage de cylindre éclairé par la gauche (Lambert sur la normale du cylindre).
- **Coupe** (vue légèrement de dessus) : cernes irréguliers (déformés par du bruit et une ondulation à 3 lobes), cœur plus sombre, fines fissures radiales, liseré d'écorce.
- **Intégration** : le tronc est teinté selon la luminosité et la dominante de couleur de la pièce, mesurées sur la photo derrière lui, et une ombre douce est dessinée derrière.

### 6.3 Peace out (`jutsus/peace_out.py`)

Imite une vidéo très compressée, comme dans le mème « bye » :
- La personne est figée au moment du signe (`Ghost`) et redessinée en **blocs de 20 px** (réduction puis agrandissement au plus proche voisin), translucide : la pièce se voit à travers.
- Toute l'image prend un coup de pixellisation pendant l'effet.
- Chaque bloc s'éteint à son propre moment ; ceux du haut (la tête) tiennent plus longtemps. Les survivants bavent sur le côté, ligne de blocs par ligne de blocs (`np.roll` par ligne), et scintillent.
- Pour ce mème, jouer le départ à l'envers est ce qui rend le mieux : l'arrivée est un rembobinage avec la personne en direct.

## 7. Webcam et caméra virtuelle

**Choisir la vraie webcam** (`camera.py`) : une fois OBS installé, macOS compte deux caméras, et OpenCV donnait l'index 0 à *OBS Virtual Camera*. GokuCam lisait sa propre sortie. Deux garde-fous :
1. `real_camera_order()` liste les caméras avec `system_profiler`, écarte les virtuelles et trie les autres par **identifiant unique**, l'ordre utilisé par OpenCV. L'ordre du JSON est différent.
2. `shows_own_output()` envoie une **mire magenta** à la caméra virtuelle et vérifie que la caméra ouverte ne la renvoie pas. Si c'est le cas, le candidat suivant est essayé. L'appel peut donc afficher un bref flash magenta au démarrage.

**Caméra virtuelle** : `pyvirtualcam` avec le backend OBS sur macOS. OBS doit seulement être installé et son extension caméra activée une fois ; il n'a pas besoin de tourner.

## 8. Performances

Mesuré sur un Apple M1 en 1280×720 :

| Étape | Temps par image |
|---|---|
| Détection + machine à états (visible ou caché) | ~20 ms |
| Effet Transmission instantanée | ~20 ms |
| Effet Substitution | ~28 ms |
| Effet Peace out | ~13 ms |
| Détection des mains sur image fixe (seulement quand un index levé est vu) | ~10 ms |
| Génération du tronc (une fois par Substitution) | ~90 ms |

Les principales astuces : détection en 640×360, fumée au 1/4 de la résolution, flous en résolution réduite, et travail propre à chaque effet (tronc, graine, bruit) fait une seule fois dans `start()` plutôt qu'à chaque image.

## 9. Vie privée

- Les images sont traitées en mémoire et envoyées uniquement à la caméra virtuelle.
- Rien n'est écrit sur le disque, sauf quand vous appuyez sur `S` : une capture de diagnostic est alors enregistrée dans `captures/`, exclu par `.gitignore`. Supprimez-le quand vous voulez.
- Le seul accès réseau est le téléchargement des trois modèles MediaPipe depuis `storage.googleapis.com` au premier lancement.
