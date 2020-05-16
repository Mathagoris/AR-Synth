# import the necessary packages
from datetime import datetime
import numpy as np
import imutils
import cv2
import time
# from synthesizer import Player, Synthesizer, Waveform
import source.camera_calibration as calib


# player = Player()
# player.open_stream()
# synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False)

recalibrate = False
cal_file = '../fiducials/calibration/'

size_of_marker = 0.0285  # side length of the marker in meter
length_of_axis = 0.05

# Load the dictionary that was used to generate the markers.
fid_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
aruco_params = cv2.aruco.DetectorParameters_create()

# Set up capture device
cap = cv2.VideoCapture(0)
time.sleep(2)
frameDict = {}

mtx = None
dist = None

fid_tracker = {}
fids_to_track = []
frames_of_history = 15


def update_fiducials(marker_ids, marker_corners):
    for fid in fids_to_track:
        position = None
        if fid in marker_ids[:, 0]:
            index = marker_ids[:, 0].index(fid)
            position = marker_corners[index][0]
        if len(fid_tracker[fid]) > frames_of_history:
            fid_tracker[fid].pop()
        fid_tracker[fid].appendleft(position)


if recalibrate:
    for i in range(5):
        print('Calibrating in ' + str(5 - i) + ' seconds...')
        time.sleep(1)
    calib.recalibrate(cal_file, cap)
else:
    calib.calibrate(cal_file)

# start looping over all the frames
while True:

    # resize the frame to have a maximum width of 400 pixels, then
    # grab the frame dimensions and construct a blob
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]

    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, fid_dict, parameters=aruco_params)
    frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    # img_undist = cv2.undistort(frame, mtx, dist, None)

    if markerIds is not None and len(markerIds) > 0:
        for i in range(len(markerIds)):
            if markerIds[i][0] == 1:
                note = (markerCorners[i][0][0] / h) * 2000 + 20
                # player.play_wave(synthesizer.generate_constant_wave(note, 3.0))
                print('marker 1: ' + str(markerCorners[i]))
    #     im = cv2.aruco.drawDetectedMarkers(img_undist, markerCorners, markerIds)
    #
    #     rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(markerCorners, size_of_marker, mtx, dist)
    #
    #     # draw axis for each marker
    #     for i in range(len(markerIds)):
    #         cv2.aruco.drawAxis(im, mtx, dist, rvecs[i], tvecs[i], length_of_axis)

    # draw the sending device name on the frame
    # cv2.putText(frame, rpiName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Display the resulting frame
    cv2.imshow('AR_SYNTH', frame)

    # draw the object count on the frame
    # label = ", ".join("{}: {}".format(obj, count) for (obj, count) in objCount.items())
    # cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

    # update the new frame in the frame dictionary
    # frameDict[rpiName] = frame

    # detect any kepresses
    key = cv2.waitKey(1) & 0xFF

    # set the last active check time as current time
    lastActiveCheck = datetime.now()

    # if the `q` key was pressed, break from the loop
    if key == ord('q'):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
