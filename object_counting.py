# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: object_counting
# Date: 8/23/18
# Author: 😏 <smirk dot cao at gmail dot com>
import cv2
import numpy as np


class CentroidTracking(object):
    def __init__(self, file_path_=None):
        self.img = None
        self.boxes = []
        self.centroids = None
        if file_path_ is not None:
            self.img = cv2.imread(file_path_)
        else:
            self.img = np.ones((480, 480, 3), dtype="uint8")

    def select_roi(self, ):
        boxes_ = []
        while True:
            roi_ = cv2.selectROI("bounding box", self.img)
            boxes_.append(roi_)
            k = cv2.waitKey(0)
            if k == 113:
                break
        print(boxes_, len(boxes_))
        cv2.destroyAllWindows()
        self.boxes = boxes_
        return boxes_

    @staticmethod
    def draw_boxes(img_, boxes_):
        for idx_, box_ in enumerate(boxes_):
            p1 = (int(box_[0]), int(box_[1]))
            p2 = (int(box_[0] + box_[2]), int(box_[1] + box_[3]))
            cv2.rectangle(img_, p1, p2, (0, 255, 255), 2, 1)
        return img_

    @staticmethod
    def get_centroid(img_, boxes_):
        gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        centroids_ = []
        if boxes_ is None:
            pass
        else:
            for idx_, box_ in enumerate(boxes_):
                left_ = int(box_[0])
                top_ = int(box_[1])
                w_ = int(box_[2])
                h_ = int(box_[3])
                sub_img_ = gray_[top_:top_+h_, left_:left_+w_]
                _, sub_img_ = cv2.threshold(sub_img_, 127, 255, 1)
                m_ = cv2.moments(sub_img_)
                if m_["m00"] != 0:
                    cx_ = int(m_["m10"] / m_["m00"])
                    cy_ = int(m_["m01"] / m_["m00"])
                else:
                    cx_, cy_ = 0, 0

                centroids_.append((left_+cx_, top_+cy_))
                print(box_, centroids_)
        return centroids_

    @staticmethod
    def draw_centroid(img_, centroids_):
        for idx_, centroid_ in enumerate(centroids_):
            cv2.circle(img_, centroid_, 5, (255, 0, 0))
            cv2.putText(img_, "centroid", tuple(x-25 for x in centroid_),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return img_


if __name__ == '__main__':
    ct = CentroidTracking()
    boxes = ct.select_roi()
    img = ct.img
    img = ct.draw_boxes(img, boxes)
    centroids = ct.get_centroid(img, boxes)
    img = ct.draw_centroid(img, centroids)
    cv2.imshow("centroid img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
