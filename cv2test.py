import cv2
import numpy as np
import numpy.lib.polynomial

imagec = cv2.imread('Line_view.png', cv2.COLOR_RGB2BGR)
imagec = cv2.resize(imagec, (150, int(150 * imagec.shape[0] / imagec.shape[1])))
image = np.array(np.apply_along_axis(lambda x: x if (x <= np.array([255, 10, 10])).all() else [255, 255, 255], 2, imagec), dtype=np.uint8)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
row_centres = []
for i in range(image.shape[0]):
	if np.any(image[i] < 100):
		rowcentre = np.median(np.argwhere(image[i] < 100))
		row_centres.append(rowcentre)

column_centres = []
image = np.swapaxes(image, 0, 1)
for i in range(image.shape[0]):
	if np.any(image[i] < 100):
		columncentre = np.median(np.argwhere(image[i] < 100))
		column_centres.append(columncentre)

for i in row_centres:
	for j in column_centres:
		cv2.circle(imagec, (round(i), round(j)), 2, (0, 0, 255), thickness=1)

# p = np.poly1d(numpy.lib.polynomial.polyfit(rows, row_centres, 3))
# for i in rows:
# 	cv2.circle(image, (round(p(i)), i), 0, (255, 255, 0), thickness=6)

cv2.imshow('', cv2.resize(imagec, (image.shape[1]*10, image.shape[0]*10)))
cv2.waitKey()
