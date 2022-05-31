import cv2
import joblib
import numpy as np
import pandas as pd
import traceback

from sklearn.decomposition import PCA

LABELS = ['flecha', 'man', 'stair', 'telephone', 'woman']
COLUMNS = ['area', 'momentx', 'momenty', 'label', 'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
           'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03', 'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12',
           'nu03']
UMBRAL_ANGLE = 0.2

class Percepcion:
    """
    Percepcion class
    """
    def __init__(self):
        self.clf_segmentation = joblib.load('./models/segmentation_model.jl')
        self.clf_marcas = joblib.load('models/rec_marcas_model.jl')
        self.clf_flechas = joblib.load('models/rec_flechas_model.jl')
        self.pca = joblib.load('./models/rec_marcas_pca.jl')
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
        paint[np.where(prediccion == 2)] = [0, 255, 0]
        return paint

    def _extract_features(self, img):
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

        return cont_list_im, ellip, lkp, des

    def procesarimagen(self, imagen):
        img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        img2 = img_hsv[:, :, 0:2]
        pred = self._etiquetar_imagen(img2)
        return self._pintar_prediccion(pred)

    def recognize_marcas(self, imagen):
        maskframe = cv2.inRange(imagen, (0, 0, 100), (50, 50, 255))
        imagen[maskframe == 0] = [255, 255, 255]
        try:
            contours, ellipse, lkeyp, dess = self._extract_features(imagen)
            if len(contours) != 1:
                contours = contours[0] if cv2.contourArea(contours[0]) > cv2.contourArea(contours[1]) else contours[1]
            M_1 = cv2.moments(contours[0])

            # if M_1["m00"] == 0: M_1["m00", "m01"] = 1
            x = int(M_1["m10"] / (M_1["m00"]+1))
            y = int(M_1["m01"] / (M_1["m00"]+1))
            data = {'momentx': x, 'momenty': y, 'area': cv2.contourArea(contours[0])}
            data.update(M_1)
            print("DATA:", data)
            odf = pd.DataFrame(columns=COLUMNS)
            odf.drop(columns=['label'], inplace=True, axis=1)
            odf = odf.append(data, ignore_index=True)
            pred_marcas = self.clf_marcas.predict(odf)
            pca = self.pca.transform(odf)
            predicted = self.clf_flechas.predict(pca)
            if predicted == 0:
                angle = ellipse[2]
                if angle < -UMBRAL_ANGLE:
                    turn = 1
                elif angle > UMBRAL_ANGLE:
                    turn = -1
                else:
                    turn = 0
                print("Turn: {}".format(turn))
                return 'flecha', turn
            return LABELS[int(pred_marcas)], 0

        except Exception as ignored:
            print(ignored)
            traceback.print_exc()
            return 'Nothing', 0

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
