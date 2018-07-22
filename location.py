# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: location
# Date: 7/20/18
# Author: Smirk <smirk dot cao at gmail dot com>
import cv2 as cv
import numpy as np
import warnings
import logging
import os


class Location(object):
    def __init__(self):
        self.image = None
        self.bouding_rect_ = None
        self.output_prefix_ = "out"
        self.output_dir = "./Output/"
        self.input_dir = "./Input/"

    def set_roi(self,
                start_x=0,
                start_y=0,
                end_x=0,
                end_y=0):
        self.start_x_ = start_x
        self.start_y_ = start_y
        self.end_x_ = end_x
        self.end_y_ = end_y
        self.loc_ = None

    def get_roi_image(self):
        if self.start_x_+self.end_x_+self.start_y_+self.end_y_ == 0:
            return self.image
        return self.image[self.start_x_:self.end_x_, self.start_y_:self.end_y_].copy()

    def read_image(self,
                   filename_=None):
        if filename_ is None:
            return None
        self.image = cv.imread(filename_)
        return self.image

    def process_image(self,
                      filename_):

        self.read_image(filename_= self.input_dir + filename_)
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
        box[:, 0] += self.start_y_
        box[:, 1] += self.start_x_

        # based on image
        self.loc_ = box
        self.draw_border()
        self.put_desc()
        self.write_image(filename_=filename_)

    def write_image(self,
                    filename_="output.jpg"):
        cv.imwrite(self.output_dir + filename_, self.image)

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

    lc = Location()
    walk_dir = os.walk(lc.input_dir)
    lc.set_roi(start_x=500, start_y=1400, end_x=3800, end_y=6700)
    # (root, dirs, files)
    for _, _, files in walk_dir:
        for filename in files:
            logging.debug(filename)
            lc.process_image(filename)


