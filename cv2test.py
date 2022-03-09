
import cv2
import numpy as np
import numpy.lib.polynomial

image = cv2.imread('Line_view2.png', cv2.IMREAD_GRAYSCALE)
imagec = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
row_centres = []
rows = []
for i in range(image.shape[0]):
	if np.any(image[i] < 100):
		rowcentre = np.median(np.argwhere(image[i] < 100))
		row_centres.append(rowcentre)
		rows.append(i)
		cv2.circle(imagec, (round(rowcentre), i), 2, (0, 0, 255), thickness=1)
p = np.poly1d(numpy.lib.polynomial.polyfit(rows, row_centres, 3))
for i in rows:
	cv2.circle(imagec, (round(p(i)), i), 0, (255, 255, 0), thickness=6)

cv2.imshow('Name', imagec)
cv2.waitKey()