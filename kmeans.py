import numpy as np
import cv2
import matplotlib.pyplot as plt

frame = cv2.imread("img.png", cv2.IMREAD_COLOR)

cv2.imshow('orig', frame)

# convert to np.float32
pixel_values = frame.reshape((-1, 3))
Z = np.float32(pixel_values)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 19, 0.1)
K = 17
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((frame.shape))
cv2.imshow('kmeans', res2)

# edges = cv2.Canny(res2, 10, 200)
# cv2.imshow('edges_canny', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
