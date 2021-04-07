
# # Connexion au capteur Intel RealSense
# # Réception des images
# # Recherche d'un squelette par OpenCV
# # Calcul de la profonfeur des points
# # Envoi en osc des positions d'articulations
# # Affichage des articulations et os dans une fenêtre OpenCV
# # Enregistrement des datas envoyées dans ./blender_osc/scripts
# # dans un json nommé avec date/heure


import sys, getopt
import math
from time import time, sleep
from json import dumps
from datetime import datetime
import numpy as np
import cv2

from oscpy.client import OSCClient
import pyrealsense2 as rs

from myconfig import MyConfig


class OscClient:

    def __init__(self, **kwargs):

        self.ip = kwargs.get('ip', b'localhost')
        self.port = kwargs.get('port', 8003)
        self.all_data = []
        self.client = OSCClient(self.ip, self.port)

    def send_message(self, points3D, bodyId=110):
        # Envoi du point en OSC en 3D
        # Liste de n°body puis toutes les coordonnées sans liste de 3
        # oscpy n'envoie pas de liste de listes

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

    def save(self):
        dt_now = datetime.now()
        dt = dt_now.strftime("%Y_%m_%d_%H_%M")
        fichier = f"./json/cap_{dt}.json"
        with open(fichier, "w") as fd:
            fd.write(dumps(self.all_data))
            print(f"{fichier} enregistré.")
        fd.close()


class SkeletonOpenCV:

    def __init__(self, **kwargs):

        self.kwargs = kwargs

        self.osc_client = OscClient(**kwargs)

        mode = kwargs.get('mode', None)
        calc = kwargs.get('calc', None)

        self.protoFile, self.weightsFile, self.num_points, self.pose_pairs\
                                                    = get_caffe_config(mode)

        self.num_points = self.num_points
        self.pose_pairs = self.pose_pairs

        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        if calc == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")

        elif calc == "gpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")
        self.set_pipeline()

        # 0.1 0.05 3 184 184 0.3 0.00392
        self.threshold = kwargs.get('threshold', None)
        self.median = kwargs.get('median', )
        self.kernel = kwargs.get('kernel', )
        self.width = kwargs.get('width', )
        self.height = kwargs.get('height', )
        self.mean = kwargs.get('mean', )
        self.scale = kwargs.get('scale', None)/255

        self.t0 = time()
        self.num = 0

        self.boucle = 1
        self.reglage_img = np.zeros((100, 600, 3), np.uint8)

    def set_pipeline(self):
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
        unaligned_frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        frames = self.align.process(unaligned_frames)
        depth = frames.get_depth_frame()
        depth_intrinsic = depth.profile.as_video_stream_profile().intrinsics

        self.depth_intrinsic = depth_intrinsic

    def create_windows(self):
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Reglage', cv2.WINDOW_AUTOSIZE)
        self.reglage_img = np.zeros((100, 600, 3), np.uint8)
        cv2.imshow('Reglage', self.reglage_img)

    def create_trackbar(self):

        cv2.createTrackbar('threshold', 'Reglage', 1, 100, self.onChange_threshhold)
        cv2.createTrackbar('width', 'Reglage', 80, 320, self.onChange_width)
        cv2.createTrackbar('height', 'Reglage', 60, 240, self.onChange_height)
        cv2.createTrackbar('kernel', 'Reglage', 1, 10, self.onChange_kernel)
        cv2.createTrackbar('mean', 'Reglage', 1, 100, self.onChange_mean)
        cv2.createTrackbar('scale', 'Reglage', 1, 100, self.onChange_scale)
        cv2.createTrackbar('median', 'Reglage', 1, 100, self.onChange_median)

    def set_init_tackbar_position(self):
        """setTrackbarPos(trackbarname, winname, pos) -> None"""
        cv2.setTrackbarPos('threshold', 'Reglage', int(self.threshold*500))
        cv2.setTrackbarPos('width', 'Reglage', int(self.width))
        cv2.setTrackbarPos('height', 'Reglage', int(self.height))
        cv2.setTrackbarPos('kernel', 'Reglage', int(self.kernel))
        cv2.setTrackbarPos('mean', 'Reglage', int(self.mean*100))
        cv2.setTrackbarPos('scale', 'Reglage', int(self.scale*50))
        cv2.setTrackbarPos('median', 'Reglage', int(self.median*1000))

    def onChange_threshhold(self, value):
        # threshold = 0.1 1, 100,
        self.threshhold = value
        self.kwargs['threshhold'] = value/500

    def onChange_width(self, value):
        # width = 160 80, 320,
        self.width = value
        self.kwargs['width'] = value

    def onChange_height(self, value):
        # height = 120 60, 240,
        self.height = value
        self.kwargs['height'] = value

    def onChange_kernel(self, value):
        # kernel = 3 1, 10,
        self.kernel = value
        self.kwargs['kernel'] = value

    def onChange_mean(self, value):
        # mean = 0.3 1, 100,
        self.mean = value
        self.kwargs['mean'] = value/100

    def onChange_scale(self, value):
        # scale = 1 1, 100,
        self.scale = value
        self.kwargs['scale'] = value/50

    def onChange_median(self, value):
        # median = 0.05 1, 100,
        self.median = value
        self.kwargs['median'] = value/1000

    def loop(self):

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # #cv2.createTrackbar('threshold', 'RealSense', 0, 100, self.onChange_threshhold)
        # #cv2.createTrackbar('width', 'RealSense', 0, 320, self.onChange_width)
        # #cv2.createTrackbar('height', 'RealSense', 0, 240, self.onChange_height)
        # #cv2.createTrackbar('kernel', 'RealSense', 0, 10, self.onChange_kernel)
        # #cv2.createTrackbar('mean', 'RealSense', 0, 100, self.onChange_mean)
        # #cv2.createTrackbar('scale', 'RealSense', 0, 100, self.onChange_scale)
        # #cv2.createTrackbar('median', 'RealSense', 0, 100, self.onChange_median)
        # #cv2.setTrackbarPos('threshold', 'RealSense', int(self.threshold*500))
        # #cv2.setTrackbarPos('width', 'RealSense', int(self.width))
        # #cv2.setTrackbarPos('height', 'RealSense', int(self.height))
        # #cv2.setTrackbarPos('kernel', 'RealSense', int(self.kernel))
        # #cv2.setTrackbarPos('mean', 'RealSense', int(self.mean*100))
        # #cv2.setTrackbarPos('scale', 'RealSense', int(self.scale*50))
        # #cv2.setTrackbarPos('median', 'RealSense', int(self.median*1000))

        # #try:
        while self.boucle:
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
            inpBlob = get_blobFromImage(frame, **self.kwargs)
            self.net.setInput(inpBlob)
            output = self.net.forward()

            # Pour ajouter tous les points en 2D et 3D, y compris None
            points2D = []
            points3D = []

            for num_point in range(self.num_points):
                # confidence map of corresponding body's part.
                probMap = output[0, self.num_points, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # Scale the point to fit on the original image
                W = output.shape[3]
                H = output.shape[2]
                x = int(((frameWidth * point[0]) / W) + 0.5)
                y = int(((frameHeight * point[1]) / H) + 0.5)

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
                    if median >= self.median:
                        # DepthIntrinsics, InputPixelAsFloat, DistanceToTargetInDepthScale)
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

            # TODO récupérer le vrai nums de body
            self.osc_client.send_message(points3D, bodyId=110)

            # Draw articulation 2D
            for point in points2D:
                if point:
                    cv2.circle(frame, (point[0], point[1]), 4, (0, 255, 255),
                                thickness=2)

            # Draw Skeleton
            for pair in self.pose_pairs:
                if points2D[pair[0]] and points2D[pair[1]]:
                    p1 = tuple(points2D[pair[0]])
                    p2 = tuple(points2D[pair[1]])
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

            cv2.imshow('RealSense', frame)
            # #cv2.imshow('Reglage', self.reglage_img)

            self.num += 1
            t = time()
            if t - self.t0 > 10:
                print("FPS =", round(self.num/10, 1))
                self.t0 = t
                self.num = 0
            if cv2.waitKey(1) == 27:
                self.boucle = 0
                break

        cv2.destroyAllWindows()

        # Pour être sûr de déconnecter le capteur, et finir le thread
        # #finally:
        self.pipeline.stop()
        self.osc_client.save()


def get_blobFromImage(frame, **kwargs):
    """
    blobFromImage   (   InputArray      image,
        double      scalefactor = 1.0,
        const Size &    size = Size(),
        const Scalar &      mean = Scalar(),
        bool    swapRB = false,
        bool    crop = false,
        int     ddepth = CV_32F )
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (width, height),
                    (0, 0, 0), swapRB=False, crop=False, ddepth=cv2.CV_32F)
    """
    # #scale = kwargs.get('scale', None)
    # #width = kwargs.get('width', None)
    # #height = kwargs.get('height', None)
    # #mean = kwargs.get('mean', None)

                                    # #1.0/255,
                                    # #(inWidth, inHeight),
                                    # #(0, 0, 0),
                                    # #swapRB=False,
                                    # #crop=False,
                                    # #ddepth=cv2.CV_32F
    # #, 1.0 / 255, (inWidth, inHeight), mean=(0, 0, 0), swapRB=False, crop=False)

    inpBlob = cv2.dnn.blobFromImage(frame,
                                    1/255,
                                    (184, 184),
                                    swapRB=False,
                                    crop = False,
                                    ddepth = cv2.CV_32F)
    return inpBlob


def get_caffe_config(mode):
    if mode == "COCO":
        protoFile = "./pose/coco/pose_deploy_linevec.prototxt"
        weightsFile = "./pose/coco/pose_iter_440000.caffemodel"
        num_points = 18
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],
                [9,10], [1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    elif mode == "MPI" :
        protoFile = "./pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "./pose/mpi/pose_iter_160000.caffemodel"
        num_points = 15
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14],
                        [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

    return protoFile, weightsFile, num_points, POSE_PAIRS


def run():
    """
    {'threshold': 0.1, 'width': 160, 'height': 120, 'kernel': 3,
    'mean': 0.3, 'scale': 1, 'mode': 'COCO', 'calc': 'cpu', 'median': 0.05,
    'ip': b'localhost', 'port': 8003}
    """

    ini_file = './rs_opencv.ini'
    my_config = MyConfig(ini_file)
    kwargs = my_config.conf['rs_opencv_osc']
    print(kwargs)
    socv = SkeletonOpenCV(**kwargs)
    sleep(1)
    socv.loop()


USAGE = """realsense_detect_skeleton.py -i <ip> -p <port> -k <kernel>
-t <threshold> -w <width> -h <height> -me <mean> -s <scale>
-mo <mode> -c <calc>

ip:             b'localhost', '192.168.1.110'
port:           8003
kernel     1 to 9
threshold       0.1
width        120 to 640
height       120 to 480
mean            0.3
scale           1
mode            'COCO'  or 'MPI'
calc            'cpu' or 'gpu'
"""

def main(argv):
    """
    i  b'localhost'
    p   8003
    k   kernel = 5
    t   threshold = 0.1
    iw   width  = 160
    ih   height = 160
    me   MEAN = 0.3
    s   SCALE = 1/255
    mo   MODE = 'COCO'  # 'MPI'  #
    c   CALC = 'cpu'
    """

    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:p:k:t:iw:ih:me:s:mo:c",["ifile=","ofile="])
    except getopt.GetoptError:
        print()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit(USAGE)
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print ('Input file is "', inputfile)
    print ('Output file is "', outputfile)


if __name__ == "__main__":
   # #main(sys.argv[1:])

    run()
