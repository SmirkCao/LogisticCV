# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: unit_test
# Date: 7/20/18
# Author: Smirk <smirk dot cao at gmail dot com>

from location import *


def test_location():
    filename = "./Input/TN193_20170531174126717_01_0000031302_NB_------------.jpg"

    lc = Location()
    lc.set_roi(start_n=500, start_m=1400, end_n=3800, end_m=6700)
    lc.process_image(filename)


if __name__ == '__main__':
    test_location()
