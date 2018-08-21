# -*-coding:utf-8-*-
# Project:  LogisticCV
# Filename: object_tracking
# Date: 8/19/18
# Author: 😏 <smirk dot cao at gmail dot com>
import cv2
from random import randint


class VideoProcessor(object):
    def __init__(self, path_, start_=-1, end_=-1):
        self.path = path_
        self.cap = None
        self.start = start_
        self.end = end_
        self.width = None
        self.height = None
        self.fps = None
        self.roi = None
        self.frame = None
        self.bboxes = None
        self.colors = None
        self.multi_tracker = None

    def load_video(self):
        self.cap = cv2.VideoCapture(self.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start*self.fps)
        return self.cap

    def select_roi(self):
        success_, frame_ = self.cap.read()
        roi_ = cv2.selectROI("select roi", frame_)
        self.frame = frame_
        self.roi = roi_
        return roi_

    @staticmethod
    def cropped_img(img_, roi_):
        return img_[int(roi_[1]):int(roi_[1] + roi_[3]), int(roi_[0]):int(roi_[0] + roi_[2])]

    def select_object(self):
        bboxes = []
        colors = []

        while True:
            bbox = cv2.selectROI('MultiTracker', self.frame)
            bboxes.append(bbox)
            colors.append((0, 255, 255))
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if k == 113:  # q is pressed
                print(k)
                break
        self.bboxes = bboxes
        self.colors = colors

        # Create MultiTracker object
        multi_tracker = cv2.MultiTracker_create()

        # Initialize MultiTracker
        for bbox in bboxes:
            multi_tracker.add(cv2.TrackerCSRT_create(), self.frame, bbox)
        self.multi_tracker = multi_tracker

    def process_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('dropout.avi', fourcc, self.fps, (self.roi[2], self.roi[3]))
        for idx in range(int(self.start*self.fps), int(self.end*self.fps)):
            print(idx)
            success_, frame_ = self.cap.read()
            # get updated location of objects in subsequent frames
            success, boxes = self.multi_tracker.update(frame_)

            # draw tracked objects
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame_, p1, p2, self.colors[i], 2, 1)
                # add label
                cv2.putText(frame_, "pacel_%d" % i,
                            (int(newbox[0]), int(newbox[1])-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
            video_writer.write(self.cropped_img(frame_, self.roi))
        video_writer.release()

    def clear(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    vp = VideoProcessor(path_="./Input/object_tracking/dropout.mp4", start_=29, end_=33)
    vp.load_video()
    vp.select_roi()
    vp.select_object()
    vp.process_video()
    vp.clear()
