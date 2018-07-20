# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: unit_test
# Date: 7/20/18
# Author: Smirk <smirk dot cao at gmail dot com>

from location import *


def test_location():
    filename = "./Input/TN193_20170531174126717_01_0000031302_NB_------------.jpg"

    lc = Location()
    lc.read_image(filename_=filename)
    lc.set_roi()
    lc.get_roi_image()
    lc.process_image()
    lc.write_image()


if __name__ == '__main__':
    test_location()
