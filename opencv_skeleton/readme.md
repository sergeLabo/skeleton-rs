# Détection d'un squelette avec OpenCV OpenVINO CUDA

## Human Pose Estimation

### Comparatif sur différentes machines

La taille des images dans cv2.dnn.blobFromImage() est size=(270, 270)

Cette valeur influe beaucoup sur le FPS obtenu. Et comme toujours en DNN, une haute qualité d'image ne garantit pas un meilleur résultat.
