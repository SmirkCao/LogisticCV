# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: location
# Date: 7/20/18
# Author: 😏 <smirk dot cao at gmail dot com>
import cv2 as cv
import numpy as np
import warnings
import logging
import glob


class Location(object):
    def __init__(self):
        self.image = None
        self.bouding_rect_ = None
        self.output_prefix_ = "out"
        self.output_dir = "./Output/"
        self.input_dir = "./Input/"
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)

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
                      image_):
        self.image = image_
        # roi_
        roi_ = self.get_roi_image()
        gray = cv.cvtColor(roi_, cv.COLOR_BGR2GRAY)
        # todo: dynamic threshold
        # constant
        (_, thresh) = cv.threshold(gray, 75, 255, cv.THRESH_BINARY)
        # OTSU not recommend for this application
        # (_, thresh) = cv.threshold(gray, 75, 255, cv.THRESH_OTSU)
        # 得到侵蚀或者膨胀的模板
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
        x_start, x_end, y_start, y_end = min(box[:, 1]), max(box[:, 1]), min(box[:, 0]), max(box[:, 0])

        return (image_[x_start:x_end, y_start:y_end]).copy()

    def write_image(self,
                    filename_="output.jpg"):
        logging.info(filename_)
        cv.imwrite(filename_, self.image)

    def get_location(self):
        if self.loc_ is None:
            #
            return None
        return self.loc_

    def draw_border(self):
        r_ = 150
        box_ = self.loc_
        cv.drawContours(self.image, [box_], -1, self.GREEN, 20, cv.LINE_4)
        # corner
        cv.circle(self.image, tuple(self.loc_[0]), r_, self.RED, cv.LINE_4)
        cv.circle(self.image, tuple(self.loc_[1]), r_, self.RED, cv.LINE_4)
        cv.circle(self.image, tuple(self.loc_[2]), r_, self.RED, cv.LINE_4)
        cv.circle(self.image, tuple(self.loc_[3]), r_, self.YELLOW, cv.LINE_4)

    def put_desc(self):
        font = cv.FONT_HERSHEY_SIMPLEX

        for idx_, loc_ in enumerate(self.loc_):
            cv.putText(self.image, str(idx_)+"-"+str(tuple(loc_)), tuple(loc_), font, 5, self.GREEN, 10)
        # parcel center
        # print(self.loc_[:, 0])
        center_x_ = (max(self.loc_[:, 0]) + min(self.loc_[:, 0]))//2
        center_y_ = (max(self.loc_[:, 1]) + min(self.loc_[:, 1]))//2
        center_ = np.array([center_x_, center_y_])
        cv.putText(self.image, "X", tuple(center_), font, 5, self.GREEN, 10)  # bottom-left corner
        # ROI center
        tmp_ = np.array([(self.start_y_ + self.end_y_) // 2, (self.start_x_ + self.end_x_) // 2])
        cv.putText(self.image, "o", tuple(tmp_), font, 5, self.BLUE, 10)  # bottom-left corner

        # adjust info
        # cv.putText(self.image, "center" + str(center_), center_, font, 5, green_, 10) # bottom-left corner
        offset_ = tmp_ - center_
        # x
        arrow_start_ = [center_[0], center_[1]+500]
        arrow_end_ = [center_[0] + np.sign(offset_[0])*200, center_[1]+500]
        cv.arrowedLine(self.image, tuple(arrow_start_),  tuple(arrow_end_), self.YELLOW, 20, tipLength=0.5)
        cv.putText(self.image, str(offset_[0]), tuple(arrow_end_), font, 5, self.YELLOW, 10)

        # y
        arrow_start_ = [center_[0], center_[1]+500]
        arrow_end_ = [center_[0], center_[1] + 500 + np.sign(offset_[1]) * 200]
        cv.arrowedLine(self.image, tuple(arrow_start_), tuple(arrow_end_), self.YELLOW, 20, tipLength=0.5)
        cv.putText(self.image, str(offset_[1]), tuple(arrow_end_), font, 5, self.YELLOW, 10)
        return offset_


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )

    lc = Location()
    lc.set_roi(start_x=500, start_y=1400, end_x=3800, end_y=6700)
    for filename in glob.iglob(lc.input_dir + "*.jpg"):
        logging.info(filename)
        image = lc.read_image(filename_=filename)
        lc.process_image(image)
        lc.draw_border()
        offset = lc.put_desc()
        lc.write_image(filename_=filename.replace(lc.input_dir, lc.output_dir))
        logging.info("offset " + str(offset))
