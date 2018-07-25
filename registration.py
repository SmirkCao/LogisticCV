# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: registration
# Date: 7/23/18
# Author: 😏 <smirk dot cao at gmail dot com>
from location import *
import cv2 as cv
import numpy as np
import glob
import logging
import warnings


class Registration():

    def __init__(self):
        self.image = None
        self.template = None
        self.bouding_rect_ = None
        self.output_prefix_ = "out"
        self.output_dir = "./Output/"
        self.input_dir = "./Input/"
        self.BLUE = (255, 0, 0)
        self.YELLOW = (0, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)

    def read_image(self,
                   filename_=None):
        if filename_ is None:
            logging.warning("no filename input")
            return None
        self.image = cv.imread(filename_)
        return self.image

    def read_template(self,
                      filename_=None):
        self.template = self.read_image(filename_)

    def write_image(self,
                    image_,
                    filename_="output.jpg"):
        cv.imwrite(filename_, image_)

    def process(self, image_):
        # 1. load image
        self.read_template("./Input/registration/template.jpg")
        # 2. scaling
        lc = Location()
        lc.set_roi(start_x=500, start_y=1400, end_x=3800, end_y=6700)
        cutted_image = lc.process_image(image_)
        logging.info(lc.loc_)
        cols_, rows_ = cutted_image.shape[:2]
        cols_th = 1000
        if cols_ > cols_th:
            sized = (cols_th, rows_*cols_th//cols_)
            logging.info(sized)
            cutted_image = cv.resize(cutted_image, sized, interpolation=cv.INTER_AREA)

        # 2. SIFT features
        gray = cv.cvtColor(self.template, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)
        des = sift.compute(gray, kp)

        img_test = cutted_image
        gray_test = cv.cvtColor(img_test, cv.COLOR_BGR2GRAY)
        sift_test = cv.xfeatures2d.SIFT_create()
        kp_test = sift_test.detect(gray_test, None)
        des_test = sift_test.compute(gray_test, kp_test)

        # 3. mathing
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        matcher = cv.FlannBasedMatcher(index_params, search_params)
        matches = matcher.match(des[1], des_test[1])
        print("Find total " + str(len(matches)) + " matches.")
        # imgMatches = None
        # imgMatches = cv.drawMatches(self.template, kp, img_test, kp_test, matches, imgMatches)
        # 4. filtered
        goodMatches = []
        minDis = 9999.0
        for i in range(0, len(matches)):
            if matches[i].distance < minDis:
                minDis = matches[i].distance
        for i in range(0, len(matches)):
            if matches[i].distance < (minDis * 4):
                goodMatches.append(matches[i])
        # imgMatches = None
        # imgMatches = cv.drawMatches(self.template, kp, img_test, kp_test, goodMatches, imgMatches)
        # 5. RANSAC
        pts_obj = []
        pts_img = []
        for i in range(0, len(goodMatches)):
            p = kp[goodMatches[i].queryIdx].pt
            pts_obj.append(p)
            pts_img.append(kp_test[goodMatches[i].trainIdx].pt)
        pts_img = np.array(pts_img).reshape(-1, 1, 2)
        pts_obj = np.array(pts_obj).reshape(-1, 1, 2)
        M_, mask = cv.findHomography(pts_img, pts_obj, cv.RANSAC, 5.0)
        goodMatches_filtered = np.array(goodMatches)[(mask == 1).flatten()]
        imgMatches = None
        imgMatches = cv.drawMatches(self.template, kp, img_test, kp_test, goodMatches_filtered, imgMatches)
        # 6. homograph
        w, h = self.template.shape[:2]
        projected_image = cv.warpPerspective(img_test, M_, (h, w))
        return projected_image


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )

    reg = Registration()

    for f in glob.iglob(reg.input_dir + "*.jpg"):
        logging.info(f)
        image = reg.read_image(filename_=f)
        desc_image = reg.process(image)

        # output
        logging.info(f.replace(reg.input_dir, reg.output_dir))
        reg.write_image(desc_image, filename_=f.replace(reg.input_dir, reg.output_dir))
