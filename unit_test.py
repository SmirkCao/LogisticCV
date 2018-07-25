# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: unit_test
# Date: 7/20/18
# Author: Smirk <smirk dot cao at gmail dot com>

from location import *
import os
import glob


def test_location():
    filename = "./Input/203_20160323141116_GR_9640010981021.jpg"

    lc = Location()
    lc.set_roi(start_x=500, start_y=1400, end_x=3800, end_y=6700)
    image = lc.read_image(filename_=filename)
    lc.process_image(image)


def test_files():
    print("*" * 10 + "glob" + "*" * 10)
    for f in glob.iglob("./Input/*.jpg"):
        print(f)
    print("*" * 10 + "os" + "*" * 10)
    for f in os.listdir("./Input/"):
        print(f)


if __name__ == '__main__':
    test_location()
    test_files()
