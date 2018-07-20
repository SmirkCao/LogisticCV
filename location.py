# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: location
# Date: 7/20/18
# Author: Smirk <smirk dot cao at gmail dot com>
import cv2 as cv
import numpy as np
import warnings


class Location(object):
    def __init__(self):
        self.image = None

    def set_roi(self,
                x1=500,
                x2=1400,
                y1=3800,
                y2=6700):
        self.x1_ = x1
        self.x2_ = x2
        self.y1_ = y1
        self.y2_ = y2

    def get_roi_image(self):
        return self.image[self.x1_:self.y1_, self.x2_:self.y2_].copy()

    def read_image(self,
                   filename_=None):
        if filename_ is None:
            return None
        self.image = cv.imread(filename_)
        return self.image

    def process_image(self):
        roi_ = self.get_roi_image()
        gray = cv.cvtColor(roi_, cv.COLOR_BGR2GRAY)
        (_, thresh) = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (60, 60))
        closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        closed = cv.erode(closed, None, iterations=2)
        closed = cv.dilate(closed, None, iterations=2)
        cnts = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        idx_ = np.argmax([len(x) for x in cnts[1]])
        c = cnts[1][idx_]
        rect = cv.minAreaRect(c)
        box = np.int0(cv.boxPoints(rect))
        cv.drawContours(self.image[self.x1_:self.y1_, self.x2_:self.y2_], [box], -1, (0, 255, 0), 20, cv.LINE_4)

    def write_image(self,
                    filename_="output.jpg"):
        cv.imwrite(filename_, self.image)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    filename = "./Input/TN193_20170531174126717_01_0000031302_NB_------------.jpg"

    lc = Location()
    lc.read_image(filename_=filename)
    lc.set_roi()
    roi_ = lc.get_roi_image()
    lc.process_image()
    lc.write_image()

