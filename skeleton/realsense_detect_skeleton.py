
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
import numpy as np
import cv2
from oscpy.client import OSCClient
import pyrealsense2 as rs
from maps import COCO_MAP, MPI_MAP
from myconfig import MyConfig


class Gestures:
    """Reconnaissance de:
        - 1 bras levé
        - 2 bras levés
        - 2 bras écartés
    """

    def __init__(self, client):
        """ MPI: points3D = [15 * [1,2,3]]"""

        self.client = client
        # Historique sur hist value
        self.hist = 50
        self.histo = []

    def add_points(self, points3D):
        """Création d'une pile de 50"""

        self.histo.append(points3D)
        if len(self.histo) > self.hist:
            del self.histo[0]
        self.gestures()

    def gestures(self):
        pass


class OscClient:
    def __init__(self, **kwargs):

        self.ip = b'localhost'  #kwargs.get('ip', None)
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

        self.mode = kwargs.get('mode', "MPI")
        self.calc = kwargs.get('calc', "cpu")

        # Création du pipeline realsense
        self.set_pipeline()

        # Création du client OSC = OSCClient(b'localhost', 8003)
        kwargs = {'ip': self.ip, 'port': self.port}
        self.osc_client = OscClient(**kwargs)

        # #self.create_trackbar()
        # #self.set_init_tackbar_position()

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
        """
        'threshold', 0.1
        'kernel', 3 : distance des pixels autour du point pour calcul profondeur
        'mean', 0.3
        'median', 0.05 :
        """
        cv2.namedWindow('Reglage', cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow('Reglage', 400, 20)
        self.black = np.zeros((400, 20, 3), dtype = "uint8")
        cv2.createTrackbar('threshold', 'Reglage', 0, 100, self.onChange_threshhold)
        cv2.createTrackbar('kernel', 'Reglage', 0, 100, self.onChange_kernel)
        cv2.createTrackbar('mean', 'Reglage', 0, 100, self.onChange_mean)
        cv2.createTrackbar('median', 'Reglage', 0, 100, self.onChange_median)

    def set_init_tackbar_position(self):
        """setTrackbarPos(trackbarname, winname, pos) -> None"""
        cv2.setTrackbarPos('threshold', 'Reglage', 50)  #int(self.threshold/0.002))  # 0.1
        cv2.setTrackbarPos('kernel', 'Reglage', int(self.kernel/10))  # 3
        cv2.setTrackbarPos('mean', 'Reglage', int(self.mean/0.006))  # 0.3
        cv2.setTrackbarPos('median', 'Reglage', int(self.median/0.001))  # 0.05

    def onChange_threshhold(self, value):
        # threshold = 0.1 0, 100
        if value == 0: value = 1
        # 0.1 = 50*k k = 0.1/50=0,002
        value *= 0.002
        self.threshhold = value

    def onChange_kernel(self, value):
        # kernel = 3 0, 100
        if value == 0: value = 1
        value = int(value/10)
        self.kernel = value

    def onChange_mean(self, value):
        # mean = 0.3 0, 100
        if value == 0: value = 1
        # 0.3 = 50*k k = 0.3/50=0,006
        value *= 0.006
        self.mean = value

    def onChange_median(self, value):
        if value == 0: value = 1
        # 0.05 = 50*k k = 0.05/50=0.001
        value *= 0.001
        self.median = value

    def loop(self):
        t0 = time()
        n = 0

        # OpenCV
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow('RealSense', 640, 480)

        try:
            while True:
                unaligned_frames = self.pipeline.wait_for_frames()

                frames = self.align.process(unaligned_frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                depth = np.asanyarray(depth_frame.get_data())
                frame = np.asanyarray(color_frame.get_data())
                frameWidth = frame.shape[1]
                frameHeight = frame.shape[0]
                inpBlob = cv2.dnn.blobFromImage(frame,
                                                scalefactor=1/255,  # pour calcul de 0 à 1
                                                size=(  self.in_width,
                                                        self.in_height),
                                                mean=self.mean,
                                                swapRB=True,
                                                crop = False,
                                                ddepth = cv2.CV_32F)
                self.net.setInput(inpBlob)
                output = self.net.forward()

                # Pour ajouter tous les points en 2D et 3D, y compris None
                points2D = []
                points3D = []

                for i in range(self.num_points):
                    # confidence map of corresponding body's part.
                    probMap = output[0, i, :, :]

                    # Find global maxima of the probMap.
                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                    # Scale the point to fit on the original image
                    x = int(((frameWidth * point[0]) / output.shape[3]) + 0.5)
                    y = int(((frameHeight * point[1]) / output.shape[2]) + 0.5)

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

                # Draw articulation 2D
                for point in points2D:
                    if point:
                        cv2.circle(frame, (point[0], point[1]), 4, (0, 255, 255),
                                    thickness=2)

                # Draw Skeleton
                for pair in self.POSE_PAIRS:
                    if points2D[pair[0]] and points2D[pair[1]]:
                        p1 = tuple(points2D[pair[0]])
                        p2 = tuple(points2D[pair[1]])
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)

                if frame.any():
                    cv2.imshow('RealSense', frame)
                # #cv2.imshow('Reglage', self.black)

                n += 1
                t = time()
                if t - t0 > 10:
                    print("FPS =", round(n/10, 1))
                    t0 = t
                    n = 0
                if cv2.waitKey(1) == 27:
                    break

            cv2.destroyAllWindows()

        finally:
            self.pipeline.stop()

        sleep(1)
        self.osc_client.save()


def run():

    ini_file = 'realsense_detect_skeleton.ini'
    my_config = MyConfig(ini_file)
    kwargs = my_config.conf['detect_skeleton']
    print(kwargs)

    skeleton = SkeletonOpenCV(**kwargs)
    sleep(1)
    skeleton.loop()


if __name__ == "__main__":
    run()
