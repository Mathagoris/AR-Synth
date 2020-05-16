from multiprocessing import Process, Pipe
from cv2 import aruco
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class Tile:
    def __init__(self, pos, num):
        self.pos = pos
        self.id = num

    def update(self, pos):
        self.pos = pos


class ActiveBox:
    def __init__(self, box, ids):
        self.box = box
        self.ids = ids

    def contains(self, tile):
        return self.box.contains(tile.pos)


class Camera:
    def __init__(self, camera, mtx=None, dist=None):
        self.camera = camera
        self.mtx = mtx
        self.dist = dist

    def read(self):
        ret, frame = self.camera.read()
        if self.mtx is not None and self.dist is not None:
            frame = cv2.undistort(frame, self.mtx, self.dist, None)
        cv2.imshow('AR_TABLE', frame)
        return frame


class Table(Process):
    def __init__(self, pipe_conn, camera=cv2.VideoCapture(0), fid_dict=aruco.Dictionary_get(aruco.DICT_6X6_250),
                 aruco_params=cv2.aruco.DetectorParameters_create(), mtx=None, dist=None):
        super(Table, self).__init__()
        self.conn = pipe_conn
        self.camera = Camera(camera, mtx, dist)
        frame = self.camera.read()
        self.table_h = frame.shape[0]
        self.table_w = frame.shape[1]
        self.fid_dict = fid_dict
        self.aruco_params = aruco_params
        self.tiles = {}
        self.active_box = self.get_boundaries()

    def get_boundaries(self):
        ret, frame = self.camera.read()
        marker_corners, marker_ids, rejected_candidates = \
            cv2.aruco.detectMarkers(frame, self.fid_dict, parameters=self.aruco_params)
        corners = []
        ids = []
        for i in range(4):
            corner = marker_corners[i]
            point = Point(corner[0][1], corner[0][0])
            corners.append(point)
            ids.append(marker_ids[i][0])
        return ActiveBox(Polygon(corners), ids)

    def run(self):
        while True:
            frame = self.camera.read()
            marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(frame, self.fid_dict, parameters=self.aruco_params)
            for i in range(len(marker_corners)):
                tile = Tile(Point(marker_corners[i][0][1], marker_corners[i][0][0]), marker_ids[i][0])
                if tile.id not in self.active_box.ids and self.active_box.contains(tile):
                    self.conn.send(tile)
