# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: location
# Date: 7/20/18
# Author: Smirk <smirk dot cao at gmail dot com>
import cv2 as cv
import numpy as np
import warnings
import logging


class Location(object):
    def __init__(self):
        self.image = None
        self.bouding_rect_ = None
        self.output_prefix_ = "out"

    def set_roi(self,
                start_n=0,
                start_m=0,
                end_n=0,
                end_m=0):
        self.start_n_ = start_n
        self.start_m_ = start_m
        self.end_n_ = end_n
        self.end_m_ = end_m
        self.loc_ = None

    def get_roi_image(self):
        if self.start_n_+self.end_n_+self.start_m_+self.end_m_ == 0:
            return self.image
        return self.image[self.start_n_:self.end_n_, self.start_m_:self.end_m_].copy()

    def read_image(self,
                   filename_=None):
        if filename_ is None:
            return None
        self.image = cv.imread(filename_)
        return self.image

    def process_image(self,
                      filename_):
        self.read_image(filename_=filename_)
        roi_ = self.get_roi_image()
        gray = cv.cvtColor(roi_, cv.COLOR_BGR2GRAY)
        # todo: dynamic threshold
        (_, thresh) = cv.threshold(gray, 50, 255, cv.THRESH_BINARY)
        #
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (60, 60))
        closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        closed = cv.erode(closed, None, iterations=2)
        closed = cv.dilate(closed, None, iterations=2)

        cnts = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        idx_ = np.argmax([len(x) for x in cnts[1]])
        c = cnts[1][idx_]
        rect = cv.minAreaRect(c)
        self.bouding_rect_ = cv.boundingRect(c)

        # based on roi_
        box = np.int0(cv.boxPoints(rect))
        box[:, 0] += self.start_m_
        box[:, 1] += self.start_n_

        # based on image
        self.loc_ = box
        lc.draw_border()
        lc.put_desc()
        lc.write_image(filename_=filename_)

    def write_image(self,
                    filename_="output.jpg"):
        cv.imwrite(filename_, self.image)

    def get_location(self):
        if self.loc_ is None:
            #
            return None
        return self.loc_

    def draw_border(self):
        r_ = 150
        green_ = (0, 255, 0)
        blue_ = (255, 0, 0)
        red_ = (0, 0, 255)
        yellow_ = (0, 255, 255)
        box_ = self.loc_
        cv.drawContours(self.image, [box_], -1, green_, 20, cv.LINE_4)
        # corner
        cv.circle(self.image, tuple(self.loc_[0]), r_, red_, cv.LINE_4)
        cv.circle(self.image, tuple(self.loc_[1]), r_, red_, cv.LINE_4)
        cv.circle(self.image, tuple(self.loc_[2]), r_, red_, cv.LINE_4)
        cv.circle(self.image, tuple(self.loc_[3]), r_, yellow_, cv.LINE_4)

    def put_desc(self):
        font = cv.FONT_HERSHEY_SIMPLEX
        green_ = (0, 255, 0)

        for idx_, loc_ in enumerate(self.loc_):
            cv.putText(self.image, str(idx_)+"-"+str(tuple(loc_)), tuple(loc_), font, 5, green_, 10)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.DEBUG)

    files = ["./Input/TN193_20170531174126717_01_0000031302_NB_------------.jpg",
             "./Input/TN097_20170531173944456_01_0000031260_NB_------------.jpg"]

    lc = Location()
    lc.set_roi(start_n=500, start_m=1400, end_n=3800, end_m=6700)
    for filename in files:
        logging.debug(filename)
        lc.process_image(filename)


