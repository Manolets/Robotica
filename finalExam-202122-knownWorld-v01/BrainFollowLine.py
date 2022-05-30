import math

import numpy as np
import pylab as p
from pyrobot.brain import Brain
import Percepcion
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class BrainFollowLine(Brain):
    NO_FORWARD = 0
    SLOW_FORWARD = 0.1
    MED_FORWARD = 0.5
    FULL_FORWARD = 1.0
    FOLLOWINGSIDE = False
    side = 1
    p = []

    NO_TURN = 0
    MED_LEFT = 0.5
    HARD_LEFT = 1.0
    MED_RIGHT = -0.5
    HARD_RIGHT = -1.0
    ORIENTING = False
    last_row = 0
    rows = []
    max_col = 0
    MAX_TURNING_TRIES = 20
    UMBRAL_OR_FLECHA = 10 # Umbral para la orientación de la flecha, ORIENTATIVO
    UMBRAL_DISTANCIA = 1

    FRONT = 0
    NOLINE = False
    SIDE = 100
    FRONT_LEFT = -1
    FRONT_RIGHT = 100
    RIGHT = 2
    AVOIDING_FRONT = False
    pre_th = 0
    pre_x = 0
    pre_y = 0
    SIDE = {1: 'right', -1: 'left'}
    FINDINGLINESTATE = 0
    TURNINGTRIES = 0

    NO_ERROR = 0
    directions = [0, 0, 0, 0, 0]

    def setup(self):
        self.perceptor = Percepcion.Percepcion()
        self.image_sub = rospy.Subscriber("/image", Image, self.callback)
        self.robot.range.units = 'ROBOTS'
        self.bridge = CvBridge()

    def callback(self, data):
        self.rosImage = data

    def destroy(self):
        cv2.destroyAllWindows()

    def follow_wall(self):
        print("following wall")
        left_dis = self.robot.range['left'][0].distance()
        right_dis = self.robot.range['right'][0].distance()
        if(left_dis < right_dis):
            if left_dis < self.UMBRAL_DISTANCIA/2:
                self.move(self.SLOW_FORWARD, self.HARD_RIGHT)
            else:
                self.move(self.SLOW_FORWARD, self.MED_LEFT)
        else:
            if right_dis < self.UMBRAL_DISTANCIA/2:
                self.move(self.SLOW_FORWARD, self.HARD_LEFT)
            else:
                self.move(self.SLOW_FORWARD, self.MED_RIGHT)


    # Obtener función de laself.find_line() línea que seguir
    def obtain_function(self, image):
        image = cv2.resize(image, (60, 60))
        image = np.array(
            np.apply_along_axis(lambda x: x if (x <= np.array([255, 10, 10])).all() else [255, 255, 255], 2, image),
            dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (320, 240))
        row_centres = []
        self.rows = []
        for i in range(image.shape[0]):
            if np.any(image[i] < 100):
                rowcentre = np.median(np.argwhere(image[i] < 100))
                row_centres.append(rowcentre)
                self.rows.append(i)

        imagec = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
        pesos = np.arange(len(row_centres))
        linePoints = np.poly1d(np.lib.polynomial.polyfit(self.rows, row_centres, 2, w=pesos))
        self.max_col = int(round(linePoints(np.max(self.rows))))
        for i in self.rows:
            cv2.circle(imagec, (int(round(linePoints(i))), i), 0, (255, 255, 0), thickness=6)
        cv2.imshow("Stage Camera Image", imagec)
        cv2.waitKey(1)
        return linePoints, image.shape[1]

    def there_obstacle(self):
        for i in range(1, 7):
            if self.robot.range[i].distance() < self.UMBRAL_DISTANCIA:
                return True
        return False

    def search_obstacle(self):
        side = {'front-left' : -1, 'front': -1, 'front-right': 1}

        angle = [0, 0, 0]
        for j, i in enumerate(side):
            min_dis = min([s.distance() for s in self.robot.range[i]])
            angle[j] = side[i] * max(0, np.cos(90*min_dis/self.UMBRAL_DISTANCIA))

        return angle[np.argmax(np.abs(angle))]

    def there_wall(self):
        return self.robot.range['right'][0].distance() < self.UMBRAL_DISTANCIA or self.robot.range['left'][0].distance() < self.UMBRAL_DISTANCIA

    def follow_line(self, imageGray, ps, d):
        # Cálculo de desviación de la línea sobre el centro para establecer parámetro de movimiento
        for i in range(imageGray.shape[0]):
            ps.append(self.p(i) - d / 2)
        mean_err = np.mean(ps) / 100
        return abs(1 - min(1, abs(mean_err))), - max(min(4, mean_err * 4), -4)

    def step(self):
        # take the last image received from the camera and convert it into
        # opencv format
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "bgr8")
        except CvBridgeError as e:
            print(e)

        # display the robot's camera's image using opencv
        # cv2.imshow("Stage Camera Image", cv_image)
        # cv2.waitKey(1)

        # write the image to a file, for debugging etc.
        # cv2.imwrite("debug-capture.png", cv_image)

        # convert the image into grayscale
        angle = 0
        if self.there_obstacle():
            angle = self.search_obstacle()
        if angle != 0:
            if abs(angle) < 0.4:
                forward = 1-abs(angle)
            else:
                forward = 0
            self.move(forward, angle)
            #print("forward ", forward)
            #print("angle ", angle)
            return

        if cv_image is not None:
            marca, orientacion = self.perceptor.recognize_marcas(cv_image.copy())
            print("Marca reconocida: ", marca)
            if marca == 'flecha' and abs(orientacion) > self.UMBRAL_OR_FLECHA:
                self.move(self.SLOW_FORWARD, orientacion)
                return

        imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # determine the robot's deviation from the line.
        d = None
        try:
            self.p, d = self.obtain_function(cv_image)
        except Exception as ignored:
            if self.there_wall():
                self.follow_wall()
            return
        ps = []
        if self.p is not None and len(self.p) > 0:  # Should be always True
            forward, turn = self.follow_line(imageGray, ps, d)
            self.move(forward, turn)
        else:
            print('line lost')
            # exit()


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine)
