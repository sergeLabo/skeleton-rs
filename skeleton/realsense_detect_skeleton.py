
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
fichier

class OscClient:

    def __init__(self, **kwargs):

        self.ip = kwargs.get('ip', None)
        self.port = kwargs.get('port', None)
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
        fichier = f"./blender_osc/scripts/cap_{dt}.json"
        with open(fichier, "w") as fd:
            fd.write(dumps(self.all_data))
            print(f"{fichier} enregistré.")
        fd.close()

class RealSensePipeLine:

    def __init__(self, **kwargs):
        """mode = 'COCO' ou 'MPI'"""

        mode = kwargs.get('mode', None)
        calc = kwargs.get('calc', None)

        cf = self.get_caffe_config(mode)
        self.protoFile = cf[0]
        self.weightsFile = cf[1]
        self.num_points = cf[2]
        self.pose_pairs = cf[3]

        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        if calc == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU device")

        elif calc == "gpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU device")
        self.set_pipe_line()

    def get_caffe_config(self, mode):
        if mode == "COCO":
            protoFile = "pose/coco/pose_deploy_linevec.prototxt"
            weightsFile = "pose/coco/pose_iter_440000.caffemodel"
            num_points = 18
            POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],
                    [9,10], [1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

        elif mode == "MPI" :
            protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
            weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
            num_points = 15
            POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14],
                            [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]

        return protoFile, weightsFile, num_points, POSE_PAIRS

    def set_pipe_line(self):
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


class SkeletonOpenCV:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.osc_client = OscClient(**kwargs)
        self.rspl = RealSensePipeLine(**kwargs)
        self.num_points = self.rspl.num_points
        self.pose_pairs = self.rspl.pose_pairs

        self.threshold = kwargs.get('threshold', None)
        self.median = kwargs.get('median', None)
        self.kernel_size = kwargs.get('kernel_size', None)

        self.data = []

        self.t0 = time()
        self.num = 0
        self.data = []  # Pour enregistrement d'un json
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    def loop(self):
        try:
            while True:
                unaligned_frames = self.rspl.pipeline.wait_for_frames()
                frames = self.rspl.align.process(unaligned_frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                depth = np.asanyarray(depth_frame.get_data())
                frame = np.asanyarray(color_frame.get_data())
                frameWidth = frame.shape[1]
                frameHeight = frame.shape[0]
                inpBlob = get_blobFromImage(frame, **self.kwargs)
                self.rspl.net.setInput(inpBlob)
                output = self.rspl.net.forward()

                H = output.shape[2]
                W = output.shape[3]

                # Pour ajouter tous les points en 2D et 3D, y compris None
                points2D = []
                points3D = []

                for num_point in range(self.num_points):
                    # confidence map of corresponding body's part.
                    probMap = output[0, self.num_points, :, :]

                    # Find global maxima of the probMap.
                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                    # Scale the point to fit on the original image
                    x = int(((frameWidth * point[0]) / W) + 0.5)
                    y = int(((frameHeight * point[1]) / H) + 0.5)

                    if prob > self.threshold :  # 0.1
                        points2D.append([x, y])
                        kernel = []
                        x_min = max(x - self.kernel_size, 0)  # mini à 0
                        x_max = max(x + self.kernel_size, 0)
                        y_min = max(y - self.kernel_size, 0)
                        y_max = max(y + self.kernel_size, 0)
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
                                                                    depth_intrinsic,
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

                self.num += 1
                t = time()
                if t - self.t0 > 10:
                    print("FPS =", round(self.num/10, 1))
                    self.t0 = t
                    self.num = 0
                if cv2.waitKey(1) == 27:
                    break

            cv2.destroyAllWindows()

        # Pour être sûr de déconnecter le capteur, et finir le thread
        finally:
            self.rspl.pipeline.stop()
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
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0/255, (in_width, in_height),
                    (0, 0, 0), swapRB=False, crop=False, ddepth=cv2.CV_32F)
    """
    scale = kwargs.get('scale', 1)/255
    in_width = kwargs.get('in_width', None)
    in_height = kwargs.get('in_height', None)
    mean = kwargs.get('mean', None)

    inpBlob = cv2.dnn.blobFromImage(frame,
                                    scalefactor=scale,
                                    size=(in_width, in_height),
                                    mean=mean,
                                    swapRB=False,
                                    crop = False,
                                    ddepth = cv2.CV_32F)
    return inpBlob


def run():
    """
    {'threshold': 0.1, 'in_width': 160, 'in_height': 120, 'kernel_size': 3,
    'mean': 0.3, 'scale': 1, 'mode': 'COCO', 'calc': 'cpu', 'median': 0.05,
    'ip': b'localhost', 'port': 8003}
    """

    ini_file = './rs_opencv.ini'
    my_config = MyConfig(ini_file)
    kwargs = my_config.conf['rs_opencv_osc']
    print(kwargs)
    socv = SkeletonOpenCV(**kwargs)
    sleep(5)
    socv.loop()


USAGE = """realsense_detect_skeleton.py -i <ip> -p <port> -k <kernel_size>
-t <threshold> -w <in_width> -h <in_height> -me <mean> -s <scale>
-mo <mode> -c <calc>

ip:             b'localhost', '192.168.1.110'
port:           8003
kernel_size     1 to 9
threshold       0.1
in_width        120 to 640
in_height       120 to 480
mean            0.3
scale           1
mode            'COCO'  or 'MPI'
calc            'cpu' or 'gpu'
"""

def main(argv):
    """
    i  b'localhost'
    p   8003
    k   kernel_size = 5
    t   threshold = 0.1
    iw   in_width  = 160
    ih   in_height = 160
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
