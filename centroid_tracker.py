# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: centroid_tracker
# Date: 8/23/18
# Author: 😏 <smirk dot cao at gmail dot com>
# refs
# 1. [Face detection with OpenCV and deep learning](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)

from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np
import cv2
import logging
import imutils


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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("tracker")

    np.random.seed(42)
    ct = CentroidTracker()
    net = cv2.dnn.readNetFromCaffe("./Input/face_detector/deploy.prototxt.txt",
                                   "./Input/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    cap = cv2.VideoCapture(0)
    while True:
        # get a frame
        _, frame = cap.read()
        logger.info("read camera")
        frame = imutils.resize(frame, width=400)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # 1. Mean subtraction
        # 2. Scaling
        # 3. And optionally channel swapping
        # show blob for understanding
        cv2.imshow("blob", np.transpose(blob[0], (1, 2, 0)))
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        logger.info("detection")
        detections = net.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            logger.info(detections)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logger.info(cap.get(cv2.CAP_PROP_FPS))
    logger.info("width %d, heigh %d" % (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    cv2.destroyAllWindows()