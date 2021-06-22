
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
print("Vérification de la source de OpenCV:", cv2.__version__)

import pyrealsense2 as rs

from myconfig import MyConfig


class OpenCVSkeleton:

    def __init__(self,  **kwargs):

        self.threshold = kwargs.get('threshold', 0.1)
        self.kernel = kwargs.get('kernel', 3)
        self.mean = kwargs.get('mean', 0.3)
        self.median = kwargs.get('median', 0.05)

        self.mode = kwargs.get('mode', "MPI")
        self.calc = kwargs.get('calc', "cpu")
        self.loop = 1

        # Skeleton blob
        self.num_points = 15
        self.POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14],
                        [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


        # Initialisation du Deep Neural Network
        self.set_dnn()

        # Création du pipeline realsense
        self.set_pipeline()

    def set_dnn(self):
        """Mode mpi. Les xml et bin sont obtenu avec OpenVINO"""

        # CaffeModel original
        self.protoFile = "pose/mpi/pose.prototxt"
        self.weightsFile = "pose/mpi/pose.caffemodel"

        # CaffeModel original faster dit light
        self.protoFile_light = "pose/MPI_light/pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile_light = "pose/MPI_light/pose_iter_160000.caffemodel"

        # Body 25
        self.protoFile_body_25 = "pose/body_25/pose_deploy.prototxt"
        self.weightsFile_body_25 = "pose/body_25/pose_iter_584000.caffemodel"

        # OpenVINO
        self.xml = "pose/openvino_mpi/pose.xml"
        self.bin = "pose/openvino_mpi/pose.bin"

        # OpenVINO light
        self.xml_light = "pose/openvino_mpi_light/pose_faster_4_stages.xml"
        self.bin_light = "pose/openvino_mpi_light/pose_faster_4_stages.bin"

        # CPU avec OpenCV                                                                   fps = 1.1
        if self.calc == "cpu":  #                                                  MSI      fps = 1.7
            self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")

        # CPU avec OpenCV et model light                                                    fps = 1.4
        elif self.calc == "cpu_light":
            self.net = cv2.dnn.readNetFromCaffe(self.protoFile_light, self.weightsFile_light)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device with Light CaffeModel")

        # CPU avec OpenCV et body_25                                                        fps = 0.4
        elif self.calc == "cpu_body_25":
            self.net = cv2.dnn.readNetFromCaffe(self.protoFile_body_25, self.weightsFile_body_25)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device with body_25")

        # CPU avec OpenVINO                                                                 fps = 1.6
        elif self.calc == "openvino_cpu":
            self.net = cv2.dnn.readNet(self.xml, self.bin)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device with OpenVINO")

        # CPU avec OpenVINO Light                                                           fps = 2.2
        elif self.calc == "openvino_cpu_light":
            self.net = cv2.dnn.readNet(self.xml_light, self.bin_light)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device with OpenVINO Light")

        # GPU                                                                     GTX1060   fps = 2.5
        elif self.calc == "gpu":
            self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")

        # GPU et model light                                                      GTX1060   fps = 3.7
        elif self.calc == "gpu_light":  #                                         MSI       fps = 2.0
            self.net = cv2.dnn.readNetFromCaffe(self.protoFile_light, self.weightsFile_light)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device with Light CaffeModel")

        # Intel NCS2 Stick Intel® Neural Compute Stick 2 (Intel® NCS2)                      fps = 1.6
        elif self.calc == "ncs2":
            self.net = cv2.dnn.readNet(self.xml, self.bin)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
            print("Using Intel NCS2 Stick device with OpenVINO")

        # Intel NCS2 Stick Intel® Neural Compute Stick 2 (Intel® NCS2) et model light       fps = 2.0
        elif self.calc == "ncs2_light":
            self.net = cv2.dnn.readNet(self.xml_light, self.bin_light)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
            print("Using Intel NCS2 Stick device with OpenVINO Light")

        else:
            print("Je ne connais pas cette configuration")
            os._exit(0)

    def set_pipeline(self):

        self.pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

        # Start
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        unaligned_frames = self.pipeline.wait_for_frames()
        frames = self.align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        self.depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

        # Vérification de la taille des images
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        print("Vérification de la taille des images: y =", img.shape[1],
                                                    "x =", img.shape[0])

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
            gauche, droite, haut, bas = 500, 300, 280, 100
            # #frame = crop_image(frame, gauche, droite, haut, bas)
            inpBlob = cv2.dnn.blobFromImage(frame,
                                            scalefactor=1/255,  # pour calcul de 0 à 1
                                            size=(270, 270),
                                            mean=self.mean,
                                            swapRB=True)

            # Définition du réseau neuronal
            self.net.setInput(inpBlob)
            # Multiplication avec la matrice de poids de 206 Mo
            # Runs forward pass to compute output of layer
            output = self.net.forward()

            # Pour ajouter tous les points en 2D et 3D, y compris None
            points2D = []
            points3D = []
            for i in range(self.num_points):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # ## Scale the point to fit on the original image
                # ## + 0.5 pour arrondi scientifique
                x = int(((frame.shape[1] * point[0]) / output.shape[3]) + 0.5)
                y = int(((frame.shape[0] * point[1]) / output.shape[2]) + 0.5)

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

            n += 1
            t = time()
            if t - t0 > 10:
                print("FPS =", round(n/10, 3))
                t0 = t
                n = 0
            if cv2.waitKey(1) == 27: # Echap
                self.loop = 0

        cv2.destroyAllWindows()
        self.pipeline.stop()


def blobFromImage():
    """
    Mat cv::dnn::blobFromImage  (   InputArray      image,
                                    double      scalefactor = 1.0,
                                    const Size &    size = Size(),
                                    const Scalar &      mean = Scalar(),
                                    bool    swapRB = false,
                                    bool    crop = false,
                                    int     ddepth = CV_32F )
    Python:
        retval = cv.dnn.blobFromImage(  image, scalefactor, size, mean,
                                        swapRB, crop, ddepth)

    Creates 4-dimensional blob from image. Optionally resizes and crops image
            from center, subtract mean values, scales values by scalefactor,
            swap Blue and Red channels.

    Parameters:
        image       input image (with 1-, 3- or 4-channels).
        size        spatial size for output image
        mean        scalar with mean values which are subtracted from channels.
                    Values are intended to be in (mean-R, mean-G, mean-B) order
                    if image has BGR ordering and swapRB is true.
        scalefactor multiplier for image values.
        swapRB      flag which indicates that swap first and last channels in
                    3-channel image is necessary.
        crop        flag which indicates whether image will be cropped after
                    resize or not
        ddepth      Depth of output blob. Choose CV_32F or CV_8U.

    if crop is true, input image is resized so one side after resize is equal to
        corresponding dimension in size and another one is equal or larger.
        Then, crop from the center is performed. If crop is false, direct resize
        without cropping and preserving aspect ratio is performed.

    Returns: 4-dimensional Mat with NCHW dimensions order.
    """
    pass


def crop_image(img, gauche, droite, haut, bas):
    """Coupe de droite à droite, ...etc
    Comprendre du coin en haut à gauche au coin en bas à droite
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    w, h = img.shape[1], img.shape[0]

    return img[haut:h - bas, gauche:w - droite]


def run():

    ini_file = 'opencv_skeleton.ini'
    my_config = MyConfig(ini_file)
    kwargs = my_config.conf['detect_skeleton']
    print(kwargs)

    skeleton = OpenCVSkeleton(**kwargs)
    sleep(1)
    skeleton.run()


if __name__ == "__main__":
    run()
