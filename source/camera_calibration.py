import numpy as np
import cv2
import os
from cv2 import aruco
import time

NUM_IMAGES = 60
DELAY = 1
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
ARUCO_BOARD = aruco.CharucoBoard_create(7, 5, 1, .8, ARUCO_DICT)


def take_images(file_path, cap):
    print('Calibrating, pulling images every ' + str(DELAY) + ' seconds...')
    counter = 0
    while True:
        time.sleep(DELAY)
        ret, frame = cap.read()
        cv2.imwrite(file_path + str(counter) + '.png', frame)
        counter += 1
        print(str(counter) + '/' + str(NUM_IMAGES) + ' images received.')
        if counter == NUM_IMAGES:
            break

    print('Finished collecting images!')
    return file_path


def recalibrate(file_path, cap):
    take_images(file_path, cap)
    return calibrate(file_path)


def calibrate(file_path):
    print('Starting pose estimation.')
    images = np.array([file_path + f for f in os.listdir(file_path) if f.endswith(".png")])
    order = np.argsort([int(p.split("/")[-1].split('.')[0]) for p in images])
    images = images[order]

    all_corners, all_ids, im_size = read_chessboards(images)
    print('Finished pose estimation!')
    print('Starting final calibration.')
    cal_info = calibrate_camera(all_corners, all_ids, im_size)
    print('Finished!')
    return cal_info


def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    all_corners = []
    all_ids = []
    decimator = 0
    gray = None

    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, ARUCO_DICT)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize=(3, 3),
                                 zeroZone=(-1, -1),
                                 criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, ARUCO_BOARD)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                all_corners.append(res2[1])
                all_ids.append(res2[2])

        decimator += 1

    im_size = gray.shape
    return all_corners, all_ids, im_size


def calibrate_camera(all_corners, all_ids, im_size):
    """
    Calibrates the camera using the detected corners.
    """

    camera_matrix_init = np.array([[1000., 0., im_size[0]/2.], [0., 1000., im_size[1]/2.], [0., 0., 1.]])

    dist_coeffs_init = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    # flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=all_corners,
                      charucoIds=all_ids,
                      board=ARUCO_BOARD,
                      imageSize=im_size,
                      cameraMatrix=camera_matrix_init,
                      distCoeffs=dist_coeffs_init,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors
