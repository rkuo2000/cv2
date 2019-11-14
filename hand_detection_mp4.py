#!/usr/bin/python3
#!--*-- coding: utf-8 --*--
from __future__ import division
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class general_pose_model(object):
    def __init__(self, modelpath):
        self.num_points = 22
        self.point_pairs = [[0,1],[1,2],[2,3],[3,4],
                            [0,5],[5,6],[6,7],[7,8],
                            [0,9],[9,10],[10,11],[11,12],
                            [0,13],[13,14],[14,15],[15,16],
                            [0,17],[17,18],[18,19],[19,20]]
        # self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.1
        self.hand_net = self.get_hand_model(modelpath)

    def get_hand_model(self, modelpath):

        prototxt   = os.path.join(modelpath, "hand/pose_deploy.prototxt")
        caffemodel = os.path.join(modelpath, "hand/pose_iter_102000.caffemodel")
        hand_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

        return hand_model

    def predict(self, img_cv2):
        img_height, img_width, _ = img_cv2.shape
        aspect_ratio = img_width / img_height

        inWidth = int(((aspect_ratio * self.inHeight) * 8) // 8)
        inpBlob = cv2.dnn.blobFromImage(img_cv2, 1.0 / 255, (inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)

        self.hand_net.setInput(inpBlob)

        output = self.hand_net.forward()

        # vis heatmaps
        #self.vis_heatmaps(img_cv2, output)

        #
        points = []
        for idx in range(self.num_points):
            probMap = output[0, idx, :, :] # confidence map.
            probMap = cv2.resize(probMap, (img_width, img_height))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > self.threshold:
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        return points

    def vis_heatmaps(self, img_cv2, net_outputs):
        plt.figure(figsize=[10, 10])

        for pdx in range(self.num_points):
            probMap = net_outputs[0, pdx, :, :]
            probMap = cv2.resize(probMap, (img_cv2.shape[1], img_cv2.shape[0]))
            plt.subplot(5, 5, pdx+1)
            plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
            plt.imshow(probMap, alpha=0.6)
            plt.colorbar()
            plt.axis("off")
        plt.show()


    def vis_pose(self, img_cv2, points):
        img_cv2_copy = np.copy(img_cv2)		
#        for idx in range(len(points)):
#            if points[idx]:
#                cv2.circle(img_cv2_copy, points[idx], 8, (0, 255, 255), thickness=-1,
#                           lineType=cv2.FILLED)
#                cv2.putText(img_cv2_copy, "{}".format(idx), points[idx], cv2.FONT_HERSHEY_SIMPLEX,
#                            1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        # Draw Skeleton
        for pair in self.point_pairs:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(img_cv2, points[partA], points[partB], (0, 255, 255), 3)
                cv2.circle(img_cv2, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.imshow('image', img_cv2) # display image
        cv2.waitKey(1)               # wait for vision
#        plt.figure(figsize=[10, 10])
#        plt.subplot(1, 2, 1)
#        plt.imshow(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
#        plt.axis("off")
#        plt.subplot(1, 2, 2)
#        plt.imshow(cv2.cvtColor(img_cv2_copy, cv2.COLOR_BGR2RGB))
#        plt.axis("off")
#        plt.show()

if __name__ == '__main__':
    print("[INFO]Pose estimation.")
#
    modelpath = "./models"
    pose_model = general_pose_model(modelpath)

    video_file = 'tsl_one_week.mp4'
    cap = cv2.VideoCapture(video_file)
    _, frame = cap.read()	
    vid_H, vid_W, vid_ch = frame.shape	
    print("video = %d x %d" % (vid_W, vid_H))
	
    framecount =0 	
    while True:
#        start = time.time()	
        _, frame = cap.read()
        if framecount%30==0: # every 30 frames detect once		
            res_points = pose_model.predict(frame)
#            print("[INFO]Model predicts time: ", time.time() - start)
            pose_model.vis_pose(frame, res_points)
        print('frame count: ', framecount)
        framecount+=1		

