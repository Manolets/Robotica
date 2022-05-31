import time

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
UMBRAL_ANGLE = 0.4


class Percepcion:
    """
    Percepcion class
    """
    def __init__(self):
        self.clf_segmentation = joblib.load('./models/segmentation_model.jl')
        self.rec_model1 = joblib.load('./models/rec_marcas_KNNmodel.jl')
        self.rec_model2 = joblib.load('./models/rec_marcas_tree_model.jl')
        self.rec_model3 = joblib.load('./models/rec_marcas_NCmodel.jl')
        self.pca = joblib.load('./models/rec_marcas_pca.jl')
        self.prediction_history_flechas = []
        self.prediction_history_marcas = []

    def _etiquetar_imagen(self, imagen):
        shape0 = imagen.shape[0]
        shape1 = imagen.shape[1]
        img = imagen.reshape(imagen.shape[0] * imagen.shape[1], 2)
        df = pd.DataFrame(data=img, columns=['h', 's'])
        predictions = self.clf_segmentation.predict(df)
        return predictions.reshape(shape0, shape1)

    def _pintar_prediccion(self, prediccion):
        paint = np.zeros((prediccion.shape[0], prediccion.shape[1], 3))
        paint[np.where(prediccion == 0)] = [255, 0, 0]
        paint[np.where(prediccion == 1)] = [0, 0, 255]
        paint[np.where(prediccion == 2)] = [255, 255, 255]
        return paint

    def pintar_marcas(self, prediccion):
        paint = np.zeros((prediccion.shape[0], prediccion.shape[1], 3))
        paint[np.where(prediccion == 0)] = [255, 255, 255]
        paint[np.where(prediccion == 1)] = [0, 0, 255]
        paint[np.where(prediccion == 2)] = [255, 255, 255]
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
        return self._pintar_prediccion(pred).astype(np.uint8), self.pintar_marcas(pred).astype(np.uint8)

    def recognize_marcas(self, imagen):
        imagen = imagen.astype(np.uint8)
        #maskframe = cv2.inRange(imagen, (0, 0, 100), (50, 50, 255))
        #imagen[maskframe == 0] = [255, 255, 255]
        try:
            #cv2.imwrite('/media/sf_Robotica/practica1Robotica-v4.3/finalExam-202122-knownWorld-v01/temp{}.png'.format(time.process_time_ns()), imagen)

            contours, ellipse, lkeyp, dess = self._extract_features(imagen)
            for cont in contours:
                cv2.drawContours(imagen, cont, -1, (np.random.randint(0, 255), np.random.randint(0, 255), 0), 3)

            cv2.ellipse(imagen, ellipse, (255, 0, 0), 3)
            cv2.drawKeypoints(imagen, lkeyp, imagen, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('Marcas', imagen)
            cv2.waitKey(1)
            if len(contours) != 1:
                contours = contours[0] if cv2.contourArea(contours[0]) > cv2.contourArea(contours[1]) else contours[1]
            M_1 = cv2.moments(contours[0])
            # if M_1["m00"] == 0: M_1["m00", "m01"] = 1
            x = int(M_1["m10"] / (M_1["m00"]+1))
            y = int(M_1["m01"] / (M_1["m00"]+1))
            data = {'momentx': x, 'momenty': y, 'area': cv2.contourArea(contours[0])}
            data.update(M_1)
            # print("DATA:", data)
            dess_df = pd.DataFrame(columns=['dess', 'label'])
            if dess is not None:
                dess_df = dess_df.append({'dess': dess, 'label': 0}, ignore_index=True)
            lista = []
            for i in range(len(dess_df)):
                lista.append(np.unpackbits(dess_df.iloc[i]['dess']))
            lista = np.array(lista)
            bbits = pd.DataFrame(lista)
            #bbits['label'] = dess_df['label']
            odf = pd.DataFrame(columns=COLUMNS)
            odf.drop(columns=['label'], inplace=True, axis=1)
            odf = odf.append(data, ignore_index=True)
            #pred_marcas = self.clf_marcas.predict(odf)
            pca = self.pca.transform(odf)
            #predicted = self.clf_flechas.predict(pca)
            pred1 = self.rec_model1.predict(pca)
            pred2 = self.rec_model2.predict(pca)
            pred3 = self.rec_model3.predict(bbits)
            print("Predicciones:", LABELS[int(pred1)], LABELS[int(pred2)], LABELS[int(pred3)])
            if pred3 == 0:
                angle = ellipse[2]
                if angle < -UMBRAL_ANGLE:
                    turn = 1
                elif angle > UMBRAL_ANGLE:
                    turn = -1
                else:
                    turn = 0
                print("Turn: {}".format(turn))
                return 'flecha', turn
            return LABELS[int(pred3)], 0

        except IndexError as ignored:
            return 'Nothing', 0
        except Exception as e:
            print(e)
            #traceback.print_exc()
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