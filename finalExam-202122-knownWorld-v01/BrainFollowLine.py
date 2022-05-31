import math

import numpy as np
import pylab as p
from pyrobot.brain import Brain
import Percepcion
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import traceback


class BrainFollowLine(Brain):
    NO_FORWARD = 0
    SLOW_FORWARD = 0.1
    MED_FORWARD = 0.5
    FULL_FORWARD = 0.7
    NORMAL_FORWARD = 0.4
    FOLLOWINGSIDE = False
    REBASING_WALL = None
    LAST_SEEN_WALL = None
    FIND_LINE_STATE = 0
    FOUND_LINE_CERTAINTY = 0
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
    UMBRAL_OR_FLECHA = 0.4 # Umbral para la orientación de la flecha, ORIENTATIVO
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
        self.segmented_image = None
        self.marcas_image = None
        self.last_preds = np.array([])

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
                self.move(self.MED_FORWARD, self.HARD_RIGHT)
            else:
                self.move(self.MED_FORWARD, self.HARD_LEFT)
        else:
            if right_dis < self.UMBRAL_DISTANCIA/2:
                self.move(self.MED_FORWARD, self.HARD_LEFT)
            else:
                self.move(self.MED_FORWARD, self.HARD_RIGHT)


    # Obtener función de laself.find_line() línea que seguir
    def obtain_function(self, image):
        image = cv2.resize(image, (60, 60))
        image = np.array(
            np.apply_along_axis(lambda x: x if (x == np.array([255, 0, 0])).all() else [255, 255, 255], 2, image),
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
            if self.robot.range[i].distance() < self.UMBRAL_DISTANCIA-0.2*self.UMBRAL_DISTANCIA:
                return True
        return False

    def find_line(self):
        if self.FIND_LINE_STATE == 0:
            if self.REBASING_WALL is not None:
                if self.REBASING_WALL == 'left':
                    self.move(0, self.HARD_LEFT)
                else:
                    self.move(0, self.HARD_RIGHT)
                self.FIND_LINE_STATE = 1
        else:
            self.move(self.FULL_FORWARD, 0)
            self.FIND_LINE_STATE = 0

    def search_obstacle(self):
        self.FOUND_LINE_CERTAINTY = 0
        angle = np.zeros(6)
        min_dis = np.zeros(6)
        side = 1
        for i in range(1, 7):
            if i >= 4:
                side = -1
            min_dis[i-1] = self.robot.range[i].distance()
            angle[i-1] = side *  max(0, np.cos(45 * min_dis[i-1] / self.UMBRAL_DISTANCIA))
        return angle[np.argmax(min_dis)]

    def there_wall(self):
        if self.robot.range['right'][0].distance() < self.UMBRAL_DISTANCIA :
            if self.REBASING_WALL == None:
                self.REBASING_WALL = 'right'
                self.LAST_SEEN_WALL = 'right'
        elif self.robot.range['right'][0].distance() < self.UMBRAL_DISTANCIA :
            if self.REBASING_WALL == None:
                self.REBASING_WALL = 'left'
                self.LAST_SEEN_WALL = 'left'
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
            self.segmented_image, self.marcas_image = self.perceptor.procesarimagen(cv_image.copy())
        except CvBridgeError as e:
            print(e)

        if self.there_obstacle() and self.REBASING_WALL is None:
            angle = self.search_obstacle()
            if abs(angle) < 0.6:
                forward = 1-abs(angle)
            else:
                forward = 0
            self.move(forward, angle)
            #print("forward ", forward)
            return
        if cv_image is not None:
            marca = self.perceptor.predict(self.marcas_image)
            orientacion = 0
            if marca != 'nothing':
                self.last_preds = np.append(self.last_preds, marca)
                #print("Marca reconocida: ", marca)
            else:
                if self.last_preds.size > 0:
                    unique, pos = np.unique(self.last_preds, return_inverse=True)  # Finds all unique elements and their positions
                    counts = np.bincount(pos)
                    maxpos = counts.argmax()
                    print("Marca reconocida: ", unique[maxpos])
                    self.last_preds = np.array([])
            if marca == 'flecha' and abs(orientacion) > self.UMBRAL_OR_FLECHA:
                self.move(self.SLOW_FORWARD, orientacion)
                return

        imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # determine the robot's deviation from the line.
        try:
            self.p, d = self.obtain_function(cv_image)
        except Exception as ignored:
            if self.there_wall():
                self.follow_wall()
            else:
                angle = self.MED_RIGHT if self.LAST_SEEN_WALL == 'right' else self.MED_LEFT
                self.move(self.MED_FORWARD, angle)

            return
        ps = []
        if self.p is not None and len(self.p) > 0:  # Should be always True
            #print("moving")
            forward, turn = self.follow_line(imageGray, ps, d)
            self.move(self.NORMAL_FORWARD, turn)
            self.FOUND_LINE_CERTAINTY = self.FOUND_LINE_CERTAINTY + 1
            if self.FOUND_LINE_CERTAINTY > 5:
                self.REBASING_WALL = None
            return
        else:
            print('line lost')
        angle =  self.MED_RIGHT if self.LAST_SEEN_WALL == 'right' else self.MED_LEFT
        self.move(self.SLOW_FORWARD, angle)
        print("Last seen wall: ", self.LAST_SEEN_WALL)

            # exit()


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine)
