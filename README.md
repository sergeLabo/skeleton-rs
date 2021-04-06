# rs-opencv

Détection de squelette avec capteur RealSense et
visualisation dans le Blender Game Engine


### Dépendances

* Debian 10 Buster
* python3.7
* oscpy
* numpy
* opencv-python
* CUDA
* Blender 2.79b

Le venv ne sert pas pour le BGE, ni CUDA
Donc il ne sert que pour tester utils.py et rs_utils.py


### Opencv et CUDA

* [Compilation de OpenCV avec CUDA](https://ressources.labomedia.org/installation_de_cuda)

### BGE

Quelques briques sont utilisées pour excécuter labomedia_once.py et labomedia_always.py
en tant que modules. Il ne faut pas modifier ces scripts.
Les autres scripts doivent être modifiés dans un EDI externe et enregistrés.

Le jeu ne doit jamais être lancé avec le Embedded Player avec "P", pour ne pas avoir
de soucis avec les threads de oscpy.

Lancer le jeu en terminal dans le dossier du blend:

```
blenderplayer blender_osc.blend
```


### Le capteur D455

* [Intel Realsense](https://ressources.labomedia.org/intel_realsense)

### Détection d'un squelette dans une image avec OpenCV

* [Pose Estimation ou Détection d'un squelette dans une image](https://ressources.labomedia.org/detection_d_un_squelette_dans_une_image)

Les fichiers .caffemodel (2 x 200 Mo) ne sont pas dans ce dépot.

Ils sont téléchargeables ici: [www.kaggle.com openpose-model](https://www.kaggle.com/changethetuneman/openpose-model)

Télécharger et coller:

"pose/coco/pose_iter_440000.caffemodel"

"pose/mpi/pose_iter_160000.caffemodel"

### TODO

Faire la version UPBGE 0.3

### Merci à

* [La Labomedia](https://ressources.labomedia.org/)
