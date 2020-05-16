import cv2
from cv2 import aruco
import matplotlib.pyplot as plt

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

fig = plt.figure()
nx = 5
ny = 4
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(ny, nx, i)
    img = aruco.drawMarker(aruco_dict, i, 700)
    plt.imshow(img, cmap=plt.get_cmap('gray'), interpolation="nearest")
    ax.axis("off")

plt.savefig("../fiducials/markers.pdf")
plt.show()

board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
imboard = board.draw((2000, 2000))
cv2.imwrite("../fiducials/chessboard.tiff", imboard)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(imboard, cmap=plt.get_cmap('gray'), interpolation="nearest")
ax.axis("off")
plt.show()
