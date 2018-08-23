# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: centroid_tracker
# Date: 8/23/18
# Author: 😏 <smirk dot cao at gmail dot com>
from collections import OrderedDict
from scipy.spatial import distance as dist
import cv2
import numpy as np


class CentroidTracker(object):
    def __init__(self, max_disappeared_=50):
        self.nxt_obj_id = 0
        self.objs = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared_

    def register(self, centroid_):
        self.objs[self.nxt_obj_id] = centroid_
        self.disappeared[self.nxt_obj_id] = 0
        self.nxt_obj_id += 1

    def unregister(self, obj_id_):
        del self.objs[obj_id_]
        del self.disappeared[obj_id_]

    def update(self, rects_):
        # rects_ from object detector, (startX, startY, endX, endY)
        # no objects, disappeared counting +1
        if len(rects_) == 0:
            for obj_id_ in self.disappeared.keys():
                self.disappeared[obj_id_] += 1
                # unregister obj while disappeared count over the limit
                if self.disappeared[obj_id_] > self.max_disappeared:
                    self.unregister(obj_id_)
            return self.objs

        # initialize an array of input centroids for the current frame
        input_centroids_ = np.zeros((len(rects_), 2), dtype="int")
        # convert rects to centroids
        for idx, (startX, startY, endX, endY) in enumerate(rects_):
            c_x = int((startX + endX) / 2.0)
            c_y = int((startY + endY) / 2.0)
            input_centroids_[idx] = (c_x, c_y)

        if len(self.objs) == 0:
            for i in range(len(input_centroids_)):
                self.register(input_centroids_[i])
        else:
            obj_ids_ = list(self.objs.keys())
            objs_ = list(self.objs.values())

            d_ = dist.cdist(np.array(objs_), input_centroids_)
            rows_ = d_.min(axis=1).argsort()
            cols_ = d_.argmin(axis=1)[rows_]

            used_rows_ = set()
            used_cols_ = set()

            for (row_, col_) in zip(rows_, cols_):
                if row_ in used_rows_ or col_ in used_cols_:
                    continue

                obj_id_ = obj_ids_[row_]
                self.objs[obj_id_] = input_centroids_[col_]
                self.disappeared[obj_id_] = 0

                used_rows_.add(row_)
                used_cols_.add(col_)

                unused_rows_ = set(range(d_.shape[0])).difference(used_rows_)
                unused_cols_ = set(range(d_.shape[1])).difference(used_cols_)

            if d_.shape[0] >= d_.shape[1]:
                for row_ in unused_rows_:
                    obj_id_ = obj_ids_[row_]
                    self.disappeared[obj_id_] += 1

                    if self.disappeared[obj_id_] > self.max_disappeared:
                        self.unregister(obj_id_)
            else:
                for col_ in unused_cols_:
                    self.register(input_centroids_[col_])

        return self.objs


if __name__ == '__main__':
    np.random.seed(42)
    ct = CentroidTracker()
    face_cascade = cv2.CascadeClassifier('./Input/face_detector/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        # get a frame
        _, frame = cap.read()
        faces = face_cascade.detectMultiScale(frame)
        rects = []
        for (x, y, w, h) in faces:
            rects.append((x, y, x+w, y+h))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        face_objs = ct.update(rects)
        for (idx, rect) in zip(face_objs.keys(), rects):
            print(idx, face_objs)
            cv2.rectangle(frame, rect[:2], rect[2:], (0, 255, 255))
            cv2.circle(frame, tuple(face_objs[idx]), 10, (0, 255, 255))
            cv2.putText(frame, "ID%d" % idx, tuple(x-25 for x in tuple(face_objs[idx])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("capture", frame)

        print(face_objs)
    cap.release()
    cv2.destroyAllWindows()