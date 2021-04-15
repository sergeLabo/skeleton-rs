
"""
 Connexion au capteur Intel RealSense
 Réception des images
 Recherche d'un squelette par OpenCV
 Calcul de la profonfeur des points
 Envoi en osc des positions d'articulations
 Affichage des articulations et os dans une fenêtre OpenCV
 Enregistrement des datas envoyées dans ./blender_osc/scripts
 dans un json nommé avec date/heure

'threshold', 0.1 : Probabilité de trouver un point
'kernel', 3 : distance des pixels autour du point pour calcul profondeur
'mean', 0.3 : Argument de cv2.dnn.blobFromImage
'median', 0.05 : Utilisé dans le calcul des profondeurs des pixels
"""


import math
from time import time, sleep
from json import dumps
from datetime import datetime
from collections import deque

import numpy as np
import cv2
from scipy.signal import savgol_filter
from oscpy.client import OSCClient
import pyrealsense2 as rs

from maps import COCO_MAP, MPI_MAP, GESTURES_UP
from myconfig import MyConfig


class Gestures:
    """Reconnaissance de gestes avec points MPI"""

    def __init__(self, client):
        """ MPI: points3D = [15 * [1,2,3]]"""

        self.client = client

        self.depth = 1
        self.mini = 0.8
        self.maxi = 1
        self.step = 1  # int de 1 à 5
        # 36 notes possibles
        self.encours = [0] * 36

        self.pile_size = 40
        self.nb_points = 15
        self.points = None

        pile_size = 40
        self.piles = []
        for i in range(self.nb_points):
            self.piles.append([])
            for j in range(3):
                self.piles[i].append(deque(maxlen=pile_size))

        # Filtre
        self.window_length = 21
        self.order = 2
        self.smooth = 1


    def add_points(self, points):
        """points = liste de 15 items, soit [1,2,3] soit None"""
        # Si pas de points, on passe
        self.points = points
        if points:
            for i in range(self.nb_points):
                if points[i]:
                    for j in range(3):  # 3
                        self.piles[i][j].append(points[i][j])
        self.get_depth_mini_maxi_current()
        self.gestures()

    def get_last_smooth_points(self):
        """Calcul de 15 points lissés, même structure que self.points
        Si la pile n'est pas remplie, retourne None pour ce point.
        Ne tiens pas compte de self.points
        """
        smooth_points = [0]*self.nb_points
        for i in range(self.nb_points):
            smooth_points[i] = []
            for j in range(3):
                if len(self.piles[i][j]) == self.pile_size:
                    three_points_smooth = savgol_filter(list(self.piles[i][j]),
                                                        self.window_length,
                                                        self.order)
                    pt = round(three_points_smooth[-1], 3)
                    smooth_points[i].append(pt)
        return smooth_points

    def get_depth_mini_maxi_current(self):
        """Moyenne de tous les z"""
        zs = []
        for point in self.points:
            if point:
                zs.append(point[2])
        if zs:
            zs = np.array(zs)
            # Profondeur courrante
            depth = np.average(zs)
        else:
            depth = 1
        # mini maxi
        if depth < self.mini:
            self.mini = depth
        if depth > self.maxi:
            self.maxi = depth

        self.deph = depth
        self.get_step()

    def get_step(self):
        """Division en 5 step de la profondeur
        step = int de 1 à 5, maxi = 2, mini = 0.5
        pas = (maxi - mini)/5
        si maxi = 2,5 mini = 0,5 pas = 0.4
        si depth = 2
        step = (2 - 0.5)/0.4 = 3.75 --> int(3.75) + 1 --> 4 --> vrai
        """

        if self.maxi - self.mini != 0:
            pas = (self.maxi - self.mini)/5
            if self.deph is not None:
                self.step = int((self.deph - self.mini)/pas) + 1
                if self.step > 5: self.step = 5
                if self.step < 1: self.step = 1

    def gestures(self):
        pts = self.points
        for key, val in GESTURES_UP.items():
            note = int(key*self.step)
            p2 = val[0]
            p1 = val[1]
            if pts[p1] and pts[p2]:
                if pts[p2][1] > pts[p1][1] + 0.1:
                    if not self.encours[note]:
                        if 0 < note and note < 36:
                            print("Envoi de:", note)
                            self.client.client.send_message(b'/note', [note])
                            self.encours[note] = 1
                if pts[p2][1] < pts[p1][1] - 0.1:
                    self.encours[note] = 0


class OscClient:
    def __init__(self, **kwargs):

        self.ip = kwargs.get('ip', None)
        self.port = kwargs.get('port', None)
        # Pour l'enregistrement d'un json à la fin de la capture
        self.all_data = []
        # Pour envoi multiple, si point différent de plus de 1%
        self.previous = None

        self.client = OSCClient(self.ip, self.port)

    def send_global_message(self, points3D, bodyId=110):
        """Envoi du point en OSC en 3D
            Liste de n°body puis toutes les coordonnées sans liste de 3
            oscpy n'envoie pas de liste de listes
        """

        msg = []
        for point in points3D:
            if point:
                for i in range(3):
                    # Envoi en int
                    msg.append(int(point[i]*1000))
            # Si pas de point ajout arbitraire de 3 fois -1000000
            # pour avoir toujours 3*18 valeurs dans la liste
            else:
                msg.extend((-1000000, -1000000, -1000000))  # tuple ou list

        # N° body à la fin
        msg.append(bodyId)
        self.all_data.append(msg)
        self.client.send_message(b'/points', msg)

    def send_multiples_messages(self, points3D, mode):
        """Un message par keypoint"""
        for point in points3D:
            if point and self.previous:
                new = 0
                index = points3D.index(point)
                if self.previous[index]:
                    # point = liste de 3, pt arrondi avec *100
                    pt = [int(coord*100) for coord in point]
                    previous_pt = [int(coord*100) for coord in self.previous[index]]
                    for i in range(3):
                        if abs(pt[i] - previous_pt[i]) > 10:
                            # Nouveau point éloigné du précédent
                            new = 1
                            break
                    if new:
                        print(pt)
                        if mode == 'COCO':
                            self.client.send_message(COCO_MAP[index], pt)
                        elif mode == 'MPI':
                            self.client.send_message(MPI_MAP[index], pt)
        self.previous = points3D

    def save(self):
        dt_now = datetime.now()
        dt = dt_now.strftime("%Y_%m_%d_%H_%M")
        fichier = f"./json/cap_{dt}.json"
        with open(fichier, "w") as fd:
            fd.write(dumps(self.all_data))
            print(f"{fichier} enregistré.")
        fd.close()


class SkeletonOpenCV:

    def __init__(self,  **kwargs):

        self.ip = kwargs.get('ip', '192.168.1.101')
        self.port = kwargs.get('port', 8003)

        self.in_width = kwargs.get('in_width', 640)
        self.in_height = kwargs.get('in_height', 480)

        self.threshold = kwargs.get('threshold', 0.1)
        self.kernel = kwargs.get('kernel', 3)
        self.mean = kwargs.get('mean', 0.3)
        self.median = kwargs.get('median', 0.05)

        self.gauche = kwargs.get('gauche', 0)
        self.droite = kwargs.get('droite', 0)
        self.haut = kwargs.get('haut', 0)
        self.bas = kwargs.get('bas', 0)

        self.mode = kwargs.get('mode', "MPI")
        self.calc = kwargs.get('calc', "cpu")

        # Création du pipeline realsense
        self.set_pipeline()

        # Création du client OSC = OSCClient(b'localhost', 8003)
        kwargs = {'ip': self.ip, 'port': self.port}
        self.osc_client = OscClient(**kwargs)

        self.gest = Gestures(self.osc_client)

        self.slider = kwargs.get('slider', 0)
        if self.slider:
            cv2.namedWindow('Reglage', cv2.WINDOW_AUTOSIZE)
            self.black = np.zeros((20, 600, 3), dtype = "uint8")
            self.create_trackbar()

        self.loop = 1

    def set_pipeline(self):
        if self.mode == "COCO":
            self.protoFile = "pose/coco/pose_deploy_linevec.prototxt"
            self.weightsFile = "pose/coco/pose_iter_440000.caffemodel"
            self.num_points = 18
            self.POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],
                                [1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

        elif self.mode == "MPI" :
            self.protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
            self.weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
            self.num_points = 15
            self.POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14],
                            [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        if self.calc == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")
        elif self.calc == "gpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")
        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        unaligned_frames = self.pipeline.wait_for_frames()
        frames = self.align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        self.depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

        # Vérification de la taille des images
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        print("Vérification de la taille des images:", img.shape[1], "x", img.shape[0])

    def create_trackbar(self):

        cv2.createTrackbar('threshold', 'Reglage', 0, 100, self.onChange_threshold)
        cv2.createTrackbar('kernel', 'Reglage', 0, 100, self.onChange_kernel)
        cv2.createTrackbar('mean', 'Reglage', 0, 100, self.onChange_mean)
        cv2.createTrackbar('median', 'Reglage', 0, 100, self.onChange_median)

        cv2.createTrackbar('gauche', 'Reglage', 0, 100, self.onChange_gauche)
        cv2.createTrackbar('droite', 'Reglage', 0, 100, self.onChange_droite)
        cv2.createTrackbar('haut', 'Reglage', 0, 100, self.onChange_haut)
        cv2.createTrackbar('bas', 'Reglage', 0, 100, self.onChange_bas)

        cv2.setTrackbarPos('threshold', 'Reglage', int(self.threshold/0.01))  # 0.1
        cv2.setTrackbarPos('kernel', 'Reglage', int(self.kernel*10))  # 3
        cv2.setTrackbarPos('mean', 'Reglage', int(self.mean/0.01))  # 0.3
        cv2.setTrackbarPos('median', 'Reglage', int(self.median/0.01))  # 0.05

        cv2.setTrackbarPos('gauche', 'Reglage', int(self.gauche/2))
        cv2.setTrackbarPos('droite', 'Reglage', int(self.droite/2))
        cv2.setTrackbarPos('haut', 'Reglage', int(self.haut))
        cv2.setTrackbarPos('bas', 'Reglage', int(self.bas))

    def onChange_gauche(self, value):
        self.gauche = 2*int(value)

    def onChange_droite(self, value):
        self.droite = 2*int(value)

    def onChange_haut(self, value):
        self.haut = int(value)

    def onChange_bas(self, value):
        self.bas = int(value)

    def onChange_threshold(self, value):
        # threshold = 0.1 à 1 pour  0 à 100
        if value == 0: value = 1
        # 1 si 100
        value *= 0.01
        print('threshold:', value)
        self.threshold = value

    def onChange_kernel(self, value):
        # kernel = 3 0, 100
        value = int(value/10)
        if value == 0: value = 1
        print('kernel:', value)
        self.kernel = value

    def onChange_mean(self, value):
        # mean = 0.3 de 0 à 1 pour 0, 100
        if value == 0: value = 1
        # 1 si 100
        value *= 0.01
        print('mean', value)
        self.mean = value

    def onChange_median(self, value):
        if value == 0: value = 1
        # 1 si 100
        value *= 0.01
        print('median', value)
        self.median = value

    def run(self):
        t0 = time()
        n = 0

        # OpenCV
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        while self.loop:
            unaligned_frames = self.pipeline.wait_for_frames()

            frames = self.align.process(unaligned_frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # #depth = np.asanyarray(depth_frame.get_data())
            frame = np.asanyarray(color_frame.get_data())

            top = time()
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            frame_cropped = crop_image(frame, self.gauche, self.droite,
                                              self.haut, self.bas)
            inpBlob = cv2.dnn.blobFromImage(frame_cropped,
                                            scalefactor=1/255,  # pour calcul de 0 à 1
                                            size=(  self.in_width,
                                                    self.in_height),
                                            mean=self.mean,
                                            swapRB=True,
                                            crop = False,
                                            ddepth = cv2.CV_32F)

            self.net.setInput(inpBlob)
            output = self.net.forward()
            t = time()
            # #print("fin", round((t - top), 5))  # 0.7 s

            # 640 480 540 440 68 55
            # #print(frame_width, frame_height, frame_cropped.shape[1],
                    # #frame_cropped.shape[0], output.shape[3], output.shape[2])

            # Pour ajouter tous les points en 2D et 3D, y compris None
            points2D = []
            points3D = []
            for i in range(self.num_points):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # Scale the point to fit on the original image
                # + 0.5 pour arrondi scientifique
                x = int(((frame_width * point[0]) / output.shape[3]) + 0.5)
                y = int(((frame_height * point[1]) / output.shape[2]) + 0.5)

                # #frame_width 640
                # #frame_height 480
                # #frame_cropped.shape[1] 400
                # #frame_cropped.shape[0] 440
                # #output.shape[3] 50
                # #output.shape[2] 55
                # #point[0] 24
                # #point[1] 22
                # #x 307
                # #y 209
                # 640 * 24 / 50 = 307

                # #print(frame_width, frame_height, frame_cropped.shape[1],
                # #frame_cropped.shape[0], output.shape[3], output.shape[2],
                # #point[0], point[1], x, y)

                if prob > self.threshold :  # 0.1
                    points2D.append([x, y])
                    kernel = []
                    x_min = max(x - self.kernel, 0)  # mini à 0
                    x_max = max(x + self.kernel, 0)
                    y_min = max(y - self.kernel, 0)
                    y_max = max(y + self.kernel, 0)
                    for u in range(x_min, x_max):
                        for v in range(y_min, y_max):
                            kernel.append(depth_frame.get_distance(u, v))
                    # Equivaut à median si 50
                    median = np.percentile(np.array(kernel), 50)

                    pt = None
                    point_with_deph = None
                    if median >= self.median:  # 0.05:
                        # Coordonnées du point dans un repère centré sur la caméra
                        # 3D coordinate space with origin = Camera
                        point_with_deph = rs.rs2_deproject_pixel_to_point(
                                                        self.depth_intrinsic,
                                                        [x, y],
                                                        median)
                    if point_with_deph:
                        points3D.append(point_with_deph)
                    else:
                        points3D.append(None)
                else:
                    points2D.append(None)
                    points3D.append(None)

            # Envoi des points en OSC en 3D
            self.osc_client.send_global_message(points3D)
            # #self.osc_client.send_multiples_messages(points3D, self.mode)
            self.gest.add_points(points3D)

            # Draw articulation 2D
            for point in points2D:
                if point:
                    cv2.circle(frame, (point[0], point[1]), 4, (0, 255, 255),
                                thickness=2)
                    # numéro
                    i = points2D.index(point)
                    x = point[0]
                    y = point[1]
                    cv2.putText(frame, f"{i}", (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2, lineType=cv2.LINE_AA)
            # Draw Skeleton
            for pair in self.POSE_PAIRS:
                if points2D[pair[0]] and points2D[pair[1]]:
                    p1 = tuple(points2D[pair[0]])
                    p2 = tuple(points2D[pair[1]])
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

            if frame.any():
                cv2.imshow('RealSense', frame)
            if self.slider:
                cv2.imshow('Reglage', self.black)

            n += 1
            t = time()
            if t - t0 > 10:
                print("FPS =", round(n/10, 1))
                t0 = t
                n = 0
            if cv2.waitKey(1) == 27:
                self.loop = 0

        cv2.destroyAllWindows()
        self.pipeline.stop()
        sleep(1)
        self.osc_client.save()


def crop_image(img, gauche, droite, haut, bas):
    """Coupe de droite à droite, ...etc
    Comprendre du coin en haut à gauche au coin en bas à droite
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    w, h = img.shape[1], img.shape[0]

    return img[haut:h - bas, gauche:w - droite]

def run():

    ini_file = 'realsense_detect_skeleton.ini'
    my_config = MyConfig(ini_file)
    kwargs = my_config.conf['detect_skeleton']
    print(kwargs)

    skeleton = SkeletonOpenCV(**kwargs)
    sleep(1)
    skeleton.run()


if __name__ == "__main__":
    run()
