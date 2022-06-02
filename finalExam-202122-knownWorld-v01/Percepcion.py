import time

import cv2
import joblib
import numpy as np

LABELS = ['flecha', 'man', 'stair', 'telephone', 'woman']
COLUMNS = ['area', 'momentx', 'momenty', 'label', 'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
           'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12',
           'nu03']
UMBRAL_ANGLE = 0.4
CAMERA_SHAPE = (240, 320)
CENTRE_THRESHOLD = 0.3


class Percepcion:
    """
    Percepcion class
    """

    def __init__(self):
        self.clf_segmentation = joblib.load('./models/segmentation_model.jl')
        self.pca = joblib.load('./models/rec_marcas_pca.jl')
        self.woman = []
        self.man = []
        self.telephone = []
        self.stairs = []
        self.flechas = []
        self.woman_orb = []
        self.man_orb = []
        self.flechas_orb = []
        self.telephone_orb = []
        self.stairs_orb = []
        self.load_ref_images()
        self.load_reference_descriptors()
        self.prediction_history_flechas = []
        self.prediction_history_marcas = []

    def _etiquetar_imagen(self, imagen):
        shape0 = imagen.shape[0]
        shape1 = imagen.shape[1]
        img = imagen.reshape(imagen.shape[0] * imagen.shape[1], 2)
        predictions = self.clf_segmentation.predict(img)
        return predictions.reshape(shape0, shape1)

    def _pintar_prediccion(self, prediccion):
        paint = np.zeros((prediccion.shape[0], prediccion.shape[1], 3))
        paint[np.where(prediccion == 0)] = [255, 0, 0]
        paint[np.where(prediccion == 1)] = [0, 0, 255]
        paint[np.where(prediccion == 2)] = [255, 255, 255]
        return paint.astype(np.uint8)

    def pintar_marcas(self, prediccion):
        paint = np.zeros((prediccion.shape[0], prediccion.shape[1], 3))
        paint[np.where(prediccion == 0)] = [255, 255, 255]
        paint[np.where(prediccion == 1)] = [0, 0, 255]
        paint[np.where(prediccion == 2)] = [255, 255, 255]
        return paint.astype(np.uint8)

    def load_ref_images(self):
        self.woman = [
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-1.png')))[1],
            self.procesarimagen(cv2.imread('marcas-capturasStage/woman-2.png'))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-3.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-4.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-5.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-6.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-7.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-8.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-9.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/woman-10.png')))[1],
        ]


        self.telephone = [
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-1.png')))[1],
            self.procesarimagen(cv2.imread('marcas-capturasStage/telephone-2.png'))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-3.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-4.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-5.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-6.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-7.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-8.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-9.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/telephone-10.png')))[1],
        ]

        self.stairs = [
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-1.png')))[1],
            self.procesarimagen(cv2.imread('marcas-capturasStage/stairs-2.png'))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-3.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-4.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-5.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-6.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-7.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-8.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-9.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/stairs-10.png')))[1],
        ]

        self.man = [
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-1.png')))[1],
            self.procesarimagen(cv2.imread('marcas-capturasStage/man-2.png'))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-3.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-4.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-5.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-6.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-7.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-8.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-9.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/man-10.png')))[1],
        ]

        self.flechas = [self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha1.png')))[1],
            self.procesarimagen(cv2.imread('marcas-capturasStage/Flecha2.png'))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha3.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha4.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha5.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha6.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha7.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha8.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha9.png')))[1],
            self.procesarimagen(cv2.imread(('marcas-capturasStage/Flecha10.png')))[1], ]

    def load_reference_descriptors(self):
        self.flechas_orb = []
        for img in self.flechas:
            try:
                c, _, _, _ = self.compute(img)
                if c is not None:
                    self.flechas_orb.append(c)
            except:
                pass
        self.man_orb = []
        for img in self.man:
            try:
                c, _, _, _ = self.compute(img)
                if c is not None:
                    self.man_orb.append(c)
            except:
                pass
        self.telephone_orb = []
        for img in self.telephone:
            try:
                c, _, _, _ = self.compute(img)
                if c is not None:
                    self.telephone_orb.append(c)
            except:
                pass
        self.woman_orb = []
        for img in self.woman:
            try:
                c, _, _, _ = self.compute(img)
                if c is not None:
                    self.woman_orb.append(c)
            except:
                pass
        self.stairs_orb = []
        for img in self.stairs:
            try:
                c, _, _, _ = self.compute(img)
                if c is not None:
                    self.stairs_orb.append(c)
            except:
                pass

    def spaghetty(self, contours, centre):
        npcontours = np.array(contours[0])
        distancesa = npcontours[np.array(contours[0])[:, :, 0] < centre[0]]
        distancesb = npcontours[np.array(contours[0])[:, :, 0] > centre[0]]
        areaa = cv2.contourArea(distancesa)
        areab = cv2.contourArea(distancesb)
        return 1 if areaa > areab else -1

    def compute(self, img):
        img = cv2.cvtColor(img.copy(), cv2.COLOR_RGBA2GRAY)
        img = img - 255

        cont_list_im, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # cont_list_im = np.array(cont_list_im)
        # cont_list_im = cont_list_im[np.array([cv2.contourArea(conto) for conto in cont_list_im]) > 10]
        orb = cv2.ORB_create()
        ellip = cv2.fitEllipse(cont_list_im[0])
        cen, ejes, angulo = np.array(ellip[0]), np.array(ellip[1]), ellip[2]
        kp = cv2.KeyPoint(cen[0], cen[1], np.mean(ejes) * 1.3, angulo - 90)
        lkp, des = orb.compute(img, [kp])
        if angulo >45 and angulo <135:
            return des, cen, angulo, self.spaghetty(cont_list_im, cen)
        return des, cen, angulo, None



    def hammingDist(self, d1, d2):
        assert d1.dtype == np.uint8 and d2.dtype == np.uint8
        d1_bits = np.unpackbits(d1)
        d2_bits = np.unpackbits(d2)
        return np.bitwise_xor(d1_bits, d2_bits).sum()

    def procesarimagen(self, imagen):
        img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        img2 = img_hsv[:, :, 0:2]
        pred = self._etiquetar_imagen(img2)
        return self._pintar_prediccion(pred), self.pintar_marcas(pred)

    def predict(self, pred):
        if (pred is not None):
            cv2.imshow("marcas ", pred)
        try:
            pred_computed, centre, angle, carbonara = self.compute(pred)
            if centre[1] > CAMERA_SHAPE[0]*(1-CENTRE_THRESHOLD):
                return 'nothing', -1, -1, None
        except Exception as e:
            return 'nothing', -1, -1, None

        if (pred is not None):
            cv2.circle(pred, (int(centre[0]), int(centre[1])), 5, (0, 255, 0), -1)
            cv2.imshow("marcas ", pred)
        try:
            dist = []
            for flecha in self.flechas_orb:
                dist.append(self.hammingDist(flecha, pred_computed))
            dist2 = []
            for man in self.man_orb:
                dist2.append(self.hammingDist(man, pred_computed))
            dist3 = []
            for woman in self.woman_orb:
                dist3.append(self.hammingDist(woman, pred_computed))
            dist4 = []
            for stair in self.stairs_orb:
                dist4.append(self.hammingDist(stair, pred_computed))
            dist5 = []
            for telephone in self.telephone_orb:
                dist5.append(self.hammingDist(telephone, pred_computed))

            mean1 = np.mean(dist)
            mean2 = np.mean(dist2)
            mean3 = np.mean(dist3)
            mean4 = np.mean(dist4)
            mean5 = np.mean(dist5)

            if mean1 < mean2 and mean1 < mean3 and mean1 < mean4 and mean1 < mean5:
                return 'flecha', centre, angle, carbonara
            elif mean2 < mean1 and mean2 < mean3 and mean2 < mean4 and mean2 < mean5:
                return 'man', -1, -1, None
            elif mean3 < mean1 - 0.2 * mean1 and mean3 < mean2 - 0.2 * mean2 and mean3 < mean4 - 0.2 * mean4 and mean3 < mean5 - 0.2 * mean5:
                return 'woman', -1, -1, None
            elif mean4 < mean1 and mean4 < mean2 and mean4 < mean3 and mean4 < mean5:
                if mean4 < mean1 - 0.2 * mean1:
                    return 'stairs', -1, -1, None
                else:
                    return 'flecha', centre, angle, carbonara
            elif mean5 < mean1 and mean5 < mean2 and mean5 < mean3 and mean5 < mean4:
                return 'telephone', -1, -1, None
        except IndexError:
            return 'nothing', -1, -1, None
        except Exception as e:
            pass
        return 'nothing', -1, -1, None

    def analyze_scene(self, image):
        ret = ""
        mask = cv2.inRange(image, (0, 0, 200), (255, 255, 255))
        image[mask != 0] = [0, 255, 0]

        image[:, 0:4, :] = [255, 0, 0]
        image[:, -4:, :] = [255, 0, 0]
        image[0:4, :, :] = [255, 0, 0]
        image[-4:, :, :] = [255, 0, 0]

        # cv2.imshow('Video', frame)
        blurr = cv2.GaussianBlur(image, (25, 25), 0)
        can_edges = cv2.Canny(blurr, 100, 200)
        # can_edges[:,0] = 0
        # can_edges[:,-1] = 0
        # can_edges[0,:] = 0
        # can_edges[-1,:] = 0

        contours, hier = cv2.findContours(can_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.array(contours)
        contours = contours[np.array([cv2.arcLength(c, False) for c in contours]) > 200]
        contours = contours[np.array([cv2.contourArea(c) for c in contours]) > 0]
        cv2.drawContours(image, contours, -1, (255, 255, 255), 4)

        if len(contours) == 2:
            M = cv2.moments(contours[0])
            cX0 = int(M["m10"] / (M["m00"] + 1))
            cY0 = int(M["m01"] / (M["m00"] + 1))

            M = cv2.moments(contours[1])
            cX1 = int(M["m10"] / (M["m00"] + 1))
            cY1 = int(M["m01"] / (M["m00"] + 1))
            order = cY0 < cY1
            bigger = cv2.contourArea(contours[0]) > cv2.contourArea(contours[1]) + 600
            smaler = cv2.contourArea(contours[0]) < cv2.contourArea(contours[1]) - 600

            if bigger and order or smaler and not order:
                ret = 'curvad'
            if smaler and order or bigger and not order:
                ret = 'curvai'
            else:
                ret = 'recta'
        elif len(contours) == 3:
            contours = sorted(contours, key=lambda c: cv2.contourArea(c))

            M = cv2.moments(contours[0])
            cX0 = int(M["m10"] / (M["m00"] + 1))
            cY0 = int(M["m01"] / (M["m00"] + 1))

            M = cv2.moments(contours[1])
            cX1 = int(M["m10"] / (M["m00"] + 1))
            cY1 = int(M["m01"] / (M["m00"] + 1))

            M = cv2.moments(contours[2])
            cX2 = int(M["m10"] / (M["m00"] + 1))
            cY2 = int(M["m01"] / (M["m00"] + 1))

            if cY2 > cY1 and cY2 > cY0:
                ret = 'interseccionT'
            else:
                ret = 'bifurcacion'
        elif len(contours) == 4:
            ret = 'interseccionT'
        return ret
