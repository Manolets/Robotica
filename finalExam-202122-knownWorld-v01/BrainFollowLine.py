import numpy as np
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

    # Obtener función de la línea que seguir
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

    def check_and_rebase(self):
        UMBRAL = 0.7
        self.FOLLOWINGSIDE = False  # Si se está siguiendo la 'pared' del objeto
        # TURNINGINITIATED = Si el giro que se va a realizar es el inicial para rebasar un objeto
        # Si hay linea y el giro a realizarse (en caso de girar) es el inicial, no se percibe obstáculo
        if not self.NOLINE and self.ORIENTING:
            return False

        # Se establece hacia donde girar de haber un obstáculo
        # Si lo hay en front-left, se gira a la derecha, default izquierda
        if min([s.distance() for s in
                self.robot.range['front-left']]) < UMBRAL and self.AVOIDING_FRONT and not self.ORIENTING:
            self.side = -1
            self.AVOIDING_FRONT = True
        elif not self.AVOIDING_FRONT and not self.ORIENTING:  # AVOIDINGFRONT = Si se detecta obstaculo a rebasar
            self.side = 1

        # FASES DE OBSTACULOS
        # FASE 1: Obstaculo en front: giro hacia el lado contrario al objeto
        if min([s.distance() for s in self.robot.range['front']]) < UMBRAL:
            self.move(0, 0.4 * self.side)
            self.AVOIDING_FRONT = True
            self.ORIENTING = False

        # FASE 2: Rebsar esquina del objeto
        elif min([s.distance() for s in self.robot.range['front-' + self.SIDE.get(self.side)]]) < UMBRAL:
            self.move(0.1, self.side * min(
                1 / min([s.distance() for s in self.robot.range['front-' + self.SIDE.get(self.side)]]), 1))
        # FASE 3: mover hacia adelante
        elif min(
                [s.distance() for s in self.robot.range[self.SIDE.get(self.side)]]) < UMBRAL and not self.ORIENTING:
            self.move(0.5, 0)
            # Si se separe mucho del objeto se pasa al estado de orientación
            if min([s.distance() for s in self.robot.range[self.SIDE.get(self.side)]]) > UMBRAL:
                self.ORIENTING = True
        # FASE 4: REORIENTACION
        elif min([s.distance() for s in self.robot.range[self.SIDE.get(self.side)]]) > UMBRAL and self.ORIENTING:
            self.move(0.1, -0.5 * self.side)
        # FASE 5: Seguimiento de objeto: si todavía no se termina el giro
        elif self.AVOIDING_FRONT:
            self.move(0.5, -1 * self.side)
            self.FOLLOWINGSIDE = True  # Se establece que el rebase se está efectuando
        # ELSE: No se ha efectuado operacion de rebase de obstáculos
        else:
            return False
        print('STATE: REBASING OBSTACLE')
        return True

    # Cambio de dirección de giro
    def change_turning(self):
        if self.TURNINGTRIES > self.MAX_TURNING_TRIES:
            self.reset_turns_count()
        elif self.FINDINGLINESTATE == 1:
            self.FINDINGLINESTATE = 2  # LEFT
        else:
            self.FINDINGLINESTATE = 1  # DEFAULT = RIGHT

    def reset_turns_count(self):
        self.TURNINGTRIES = -1
        self.FINDINGLINESTATE = 0


    '''
    cond = Si no se encuentra la línea o se da que la línea 
    no está lo suficientemente centrada y con suficientes valores
    
    Decide hacia qué lado buscar la línea cuando 
    se da cond y no se ha conseguido una mejora de cond en la iteración actual
    return cond
    '''

    def aproach_line(self):
        cond = self.NOLINE or (np.max(self.rows) < 220) or len(self.rows) < 100
        print('log: LINE FOUND = ', not cond)
        if cond:
            # Si se visualiza la línea, se no hay mejora en esta iteración, se cambia la dirección de búsqueda
            if self.NOLINE or np.max(self.rows) > self.last_row and self.max_col < 340 / 2:
                self.change_turning()
            elif self.TURNINGTRIES > self.MAX_TURNING_TRIES:
                self.reset_turns_count()
            self.find_line()
        return cond

    def find_line(self):
        print("STATE: SEARCHING FOR LINE")
        if self.FINDINGLINESTATE == 0:  # STATE 0 = SEARCH FORWARD
            self.move(1, 0)
            self.FINDINGLINESTATE = 1
        elif self.FINDINGLINESTATE == 1:  # STATE 1 =  INITIAL TURNING DIRECTION
            self.move(0, 10 * self.side)
            self.TURNINGTRIES = self.TURNINGTRIES + 1
            self.FINDINGLINESTATE = 2
        elif self.FINDINGLINESTATE == 2:  # STATE 2 = OPPOSITE TO INITIAL TURNING DIRECTION
            self.TURNINGTRIES = self.TURNINGTRIES + 1
            self.move(0, -10 * self.side)
        else:
            self.move(1, 0)

    def follow_line(self, imageGray, ps, d):
        # Cálculo de desviación de la línea sobre el centro para establecer parámetro de movimiento
        for i in range(imageGray.shape[0]):
            ps.append(self.p(i) - d / 2)
        mean_err = np.mean(ps) / 100
        return abs(1 - min(1, abs(mean_err))), - max(min(4, mean_err * 4), -4)

    def step(self):
        # take the last image received from the camera and convert it into
        # opencv format
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
        imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # determine the robot's deviation from the line.
        try:
            self.p, d = self.obtain_function(cv_image)
        except Exception as e:
            # SI hay excepción no se encotnró la linea
            self.p = None
            self.NOLINE = True
            if self.check_and_rebase():
                return
        # En este punto la línea se conoce
        if self.NOLINE or (not self.NOLINE and self.FOLLOWINGSIDE):  # Si se ha encontrado la linea
            print('STATE: FOUND LINE')
            # Si se ha rebasado el obstáculo
            if min([s.distance() for s in self.robot.range['front-' + self.SIDE.get(self.side * -1)]]) > 2:
                self.AVOIDING_FRONT = False
                self.ORIENTING = False
            self.aproach_line()
        self.NOLINE = False
        ps = []
        if self.p is not None:  # Should be always True
            forward, turn = self.follow_line(imageGray, ps, d)
            if not self.check_and_rebase():
                if not self.aproach_line():
                    self.last_row = 0
                    print('STATE: FOLLOWING LINE')
                    self.move(forward, turn)
        else:
            self.find_line()
            # exit()


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    return BrainFollowLine('BrainFollowLine', engine)
