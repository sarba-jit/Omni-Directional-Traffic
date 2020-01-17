#!/usr/bin/python

#############################################
# module: OmniVidBeeCount.py
# description: detect takes moving bees in a video
# moving an any direction. 
# author: vladimir kulyukin and sarbajit mukherjee
#############################################

# import the necessary packages
import csv
import datetime
import glob
import shutil
from collections import defaultdict
from collections import namedtuple

import numpy as np
import argparse
import cv2
import os

from sklearn.utils import random, shuffle
import tensorflow as tf

from Background import subtractBackground
#from BeeClassifierNet import BeeClassifierNet
import time

from beepi_convnets import beepi_convnets
from data import BEE1_IMWIDTH, BEE1_IMHEIGHT, BEE1_IMCHANNEL
from data import BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL
from data import PERSIST_BEE1_DIR
from data import PERSIST_BEE2_DIR
from clusters import clusters
# from data import VID_DIR_R_4_5_may

from Background import MOG, MOG2, KNN


CAM_HEIGHT = '1S'
ROI_SIZE_1S = 150
ROI_SIZE_2S = 90
BCKGRND = 'MOG'  # values are 'MOG', 'MOG2', 'KNN'.
ROI_SIZE = ROI_SIZE_1S if CAM_HEIGHT == '1S' else ROI_SIZE_2S
LOWER_CNT_RADIUS = 3
UPPER_CNT_RADIUS = 15
SHOW_FRAME_RESIZE = (1600, 760)
DRAW_YB_ROI_COLOR = (0, 255, 0)
DRAW_NB_ROI_COLOR = (255, 0, 0)
CONVNET_NAME = ''
OVERLAP_THRESH = 0.7

class OmniVidBeeMotionCount(object):

    def __init__(self, model=None):
        self._model = model

    def _createFrameDirAndFilename(self, vid_path, frame_dir):
        h, t = os.path.split(vid_path)
        dir_name = t.split('.')[0]
        if frame_dir[-1] == '/':
            return frame_dir + dir_name + '_orig/', dir_name
        else:
            return frame_dir + '/' + dir_name + '_orig/', dir_name

    def _createRoiDirAndFilename(self, vid_path, roi_dir):
        h, t = os.path.split(vid_path)
        dir_name = t.split('.')[0]
        if roi_dir[-1] == '/':
            return roi_dir + dir_name + '_roi/', dir_name
        else:
            return roi_dir + '/' + dir_name + '_roi/', dir_name

    def _createProcessedFrameDirAndFilename(self, vid_path, frame_dir):
        h, t = os.path.split(vid_path)
        dir_name = t.split('.')[0]
        if frame_dir[-1] == '/':
            return frame_dir + dir_name + '_pfr/', dir_name
        else:
            return frame_dir + '/' + dir_name + '_pfr/', dir_name

    def _cropROI(self, frame, x, y, roi_size):
        # print 'cropROI', x, y, frame_size
        # print frame.shape
        nrows, ncols, nc = frame.shape
        sc = int(x - roi_size / 2)
        sr = int(y - roi_size / 2)
        er, ec = 0, 0
        if sc < 0:
            sc = 0
        if sr < 0:
            sr = 0
        er = int(sr + roi_size)
        if er >= nrows:
            sr = nrows - roi_size
            er = nrows
        ec = int(sc + roi_size)
        if ec >= ncols:
            sc = ncols - roi_size
            ec = ncols
        # print sr, er, sc, ec
        roi = frame[sr:er, sc:ec]
        # roi.reshape(32, 32)
        # print 'roi.shape', roi.shape
        assert (roi.shape[0] == roi_size)
        return roi

    # This function is similar to cropROI. However, this function returns a tuple of (topX, topY, bottomX, bottomY)
    # which are the coordinates of the ROI
    # frame is the original frame from which the ROI is extracted
    # x,y is the center of the detected motion regions
    # roi_size is the size of the roi, for example if you want to crop out 90x90 roi, frame_size=90
    def _getROICoordinates(self, frame, x, y, roi_size):
        nrows, ncols, nc = frame.shape
        sc = int(x - roi_size / 2)
        sr = int(y - roi_size / 2)
        er, ec = 0, 0
        if sc < 0:
            sc = 0
        if sr < 0:
            sr = 0
        er = int(sr + roi_size)
        if er >= nrows:
            sr = nrows - roi_size
            er = nrows
        ec = int(sc + roi_size)
        if ec >= ncols:
            sc = ncols - roi_size
            ec = ncols
        topX, topY, bottomX, bottomY = sc, sr, ec, er
        #return sc, sr, ec, er
        return topX, topY, bottomX, bottomY

    # This function takes in cv2 image, its width and height. Resizes it to width x height and returns
    # the processed numpy representation of the image
    def _preprocessImage(self, image, width, height):
        x = []
        if image.shape[0] > 0 and image.shape[1] > 0:
            image = cv2.resize(image, (width, height))
            x.append(image)
            x = np.array(x, dtype='float') / 255.0
            return x
        else:
            return None

    # for testing purposes
    def containsBee(self, img):
        return self._containsBee(img)

    # This function takes in the model object and an image numpy array and returns the prediction
    def _containsBee(self, img):
        processedImg = self._preprocessImage(img, 64, 64)
        if processedImg is not None:
            #prediction = self._model.predict_label(processedImg)
            prediction = self._model.predict(processedImg)
            #print('prediction={}'.format(prediction))
            prediction = np.argmax(prediction, axis=1)
            if prediction == 0:
                return True
            elif prediction == 1:
                return False
            else:
                raise Exception('_containsBee(): unknown prediction {}'.format(prediction))
        else:
            raise Exception('_containsBee(): img is None')

    def _processMotionContours(self, frame_sb, frame, frame_copy, tick, roi_size,
                               roi_dir, roi_filename,
                               lower_cnt_radius=3, upper_cnt_radius=15,
                               draw_yb_roi_flag=False, draw_nb_roi_flag=False,
                               save_yb_roi_flag=False, save_nb_roi_flag=False):
        # detect countours
        cnts = cv2.findContours(frame_sb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        #center = None
        beeMotionCount = 0
        roiCount = 0

        print('_processMotionContours')

        if len(cnts) > 0:
            for cn, c in enumerate(cnts):
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if lower_cnt_radius <= radius <= upper_cnt_radius:
                    # crop an roi_size x roi_size region from a frame
                    # note that roi is drawn from the original frame, not from its
                    # copy frame_copy; frame_copy is used for drawing lines.
                    roi = self._cropROI(frame, x, y, roi_size)
                    x1, y1, x2, y2 = self._getROICoordinates(frame, x, y, roi_size)

                    roifn = roi_filename + '_fr_' + str(tick) + '_roi_' + str(roiCount)
                    if self._containsBee(roi):
                        #print('bee in ' + roifn)
                        if save_yb_roi_flag:
                            print('saving roi in ' + roi_dir + roifn + '_yb.png')
                            cv2.imwrite(roi_dir + roifn + '_yb.png', roi)
                        if draw_yb_roi_flag:
                            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), DRAW_YB_ROI_COLOR, 2)
                        beeMotionCount += 1
                    else:
                        #print('no bee in ' + roifn)
                        if save_nb_roi_flag:
                            #print('saving roi in ' + roi_dir + roifn + '_nb.png')
                            cv2.imwrite(roi_dir + roifn + '_nb.png', roi)
                        if draw_nb_roi_flag:
                            # roi is drawn on the frame copy so that the original frame
                            # is intact
                            print('DRAW_NB_ROI_FLAG')
                            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), DRAW_NB_ROI_COLOR, 2)
                            pass
                    
                    roiCount += 1
                    
        return frame_copy, beeMotionCount

    def _printBeeMotionROIs(self, frame_sb, frame, frame_copy, tick, roi_size,
                         roi_dir, roi_filename,
                         lower_cnt_radius=3, upper_cnt_radius=15,
                         draw_yb_roi_flag=False, draw_nb_roi_flag=False,
                         save_yb_roi_flag=False, save_nb_roi_flag=False):
        # detect countours
        cnts = cv2.findContours(frame_sb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        #center = None
        #beeMotionCount = 0
        roiCount = 0
        beeMotionROIs = []
        noBeeMotionROIs = []

        if len(cnts) > 0:
            for cn, c in enumerate(cnts):
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if lower_cnt_radius <= radius <= upper_cnt_radius:
                    # crop an roi_size x roi_size region from a frame
                    # note that roi is drawn from the original frame, not from its
                    # copy frame_copy; frame_copy is used for drawing lines.
                    roi = self._cropROI(frame, x, y, roi_size)
                    x1, y1, x2, y2 = self._getROICoordinates(frame, x, y, roi_size)

                    roifn = roi_filename + '_fr_' + str(tick) + '_roi_' + str(roiCount)
                    if self._containsBee(roi):
                        beeMotionROIs.append((x1, y1, x2, y2))
                    else:
                        noBeeMotionROIs.append((x1, y1, x2, y2))

        print('Frame {}'.format(tick))
        print('BeeMotionROIs:')
        for bmr in beeMotionROIs:
            print(bmr)
        print('NoBeeMotionROIs:')
        for nbmr in noBeeMotionROIs:
            print(nbmr)

    def _countBeeMotionROIsWithOverlap(self, frame_sb, frame, frame_copy, tick, roi_size,
                                       roi_dir, roi_filename,
                                       lower_cnt_radius=3, upper_cnt_radius=15,
                                       overlap_thresh=OVERLAP_THRESH,
                                       draw_yb_roi_flag=False, draw_nb_roi_flag=False,
                                       save_yb_roi_flag=False, save_nb_roi_flag=False):
        # detect countours
        cnts = cv2.findContours(frame_sb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        #center = None
        #beeMotionCount = 0
        roiCount = 0
        beeMotionROIs = []
        noBeeMotionROIs = []
        ROI_AREA = float(roi_size**2)
        bee_clrs = clusters()
        nobee_clrs = clusters()
        ## save all motion regions in beeMotionROIs and noBeeMotionROIs.
        if len(cnts) > 0:
            for cn, c in enumerate(cnts):
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if lower_cnt_radius <= radius <= upper_cnt_radius:
                    # crop an roi_size x roi_size region from a frame
                    # note that roi is drawn from the original frame, not from its
                    # copy frame_copy; frame_copy is used for drawing lines.
                    roi = self._cropROI(frame, x, y, roi_size)
                    x1, y1, x2, y2 = self._getROICoordinates(frame, x, y, roi_size)

                    roifn = roi_filename + '_fr_' + str(tick) + '_roi_' + str(roiCount)
                    if self._containsBee(roi):
                        if save_yb_roi_flag:
                            # print('saving roi in ' + roi_dir + roifn + '_yb.png')
                            cv2.imwrite(roi_dir + roifn + '_yb.png', roi)
                        beeMotionROIs.append((x1, y1, x2, y2))
                    else:
                        noBeeMotionROIs.append((x1, y1, x2, y2))
                        if save_nb_roi_flag:
                            # print('saving roi in ' + roi_dir + roifn + '_nb.png')
                            cv2.imwrite(roi_dir + roifn + '_nb.png', roi)
                            
                    roiCount += 1

        ### This must be a clustering algorithm. Take a bmr. Find a cluster for bmr. If you
        ### cannot find a cluster for bmr, the bmr becomes its own cluster. A cluster is a set
        ### of bmrs. A collection of clusters is a set of sets. You go through the collection of
        ### clusters. If a bmr has an overlap with at least 1 bmr in the cluster, then
        ### you add that bmr to that cluster.
        #similarity_predicate = lambda bmr1, bmr2: self._rectangleOverlapArea(bmr1, bmr2)/ROI_AREA >= overlap_thresh
        sim_pred = lambda bmr1, bmr2: self._rectangleOverlapArea(bmr1, bmr2)/ROI_AREA

        # cluster beeMotionROIs
        for bmr in beeMotionROIs:
            bee_clrs.cluster(bmr, sim_pred, overlap_thresh)

        #print('Frame {}'.format(tick))            
        #print('BeeMotionROIs:')
        #for bmr in beeMotionROIs:
        #    print(bmr)            
        #print('Bee Motion Clusters:')
        #for i, clr in enumerate(bee_clrs.getClusters()):
        #    print('{}) {}'.format(i, clr))

        #print('Number of bee motions: {}'.format(len(beeMotionROIs)))
        #print('Number of true bee motions: {}'.format(len(bee_clrs.getClusters())))            

        # cluster noBeeMotionROIs
        for nbmr in noBeeMotionROIs:
            nobee_clrs.cluster(nbmr, sim_pred, overlap_thresh)
            
        #print('NoBeeMotionROIs:')
        #for nbmr in noBeeMotionROIs:
        #    print(nbmr)
        #print('No Bee Motion Clusters:')
        #for i, clr in enumerate(nobee_clrs.getClusters()):
        #    print('{}) {}'.format(i, clr))

        #print('Number of no-bee motions: {}'.format(len(noBeeMotionROIs)))
        #print('Number of true no-bee motions: {}'.format(len(nobee_clrs.getClusters())))

        if draw_yb_roi_flag:
            #print('DRAWING YB CLUSTERS')
            for bc in bee_clrs.getClusters():
                # perhaps, we can compute the average x1, y1, x2, y2.
                x1, y1, x2, y2 = bc[0]
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), DRAW_YB_ROI_COLOR, 2)

        if draw_nb_roi_flag:
            #print('DRAWING NB CLUSTERS')
            #print(len(nobee_clrs.getClusters()))
            for nc in nobee_clrs.getClusters():
                # perhaps, we can compute the average x1, y1, x2, y2.
                x1, y1, x2, y2 = nc[0]
                #print('Drawing nb cluster: {}, {}, {}, {}'.format(x1, y1, x2, y2))
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), DRAW_NB_ROI_COLOR, 2)
                
        return frame_copy, len(bee_clrs.getClusters())
        
    def _rectangleArea(self, r):
        """
        r is a 4-tuple (topX, topY, bottomX, bottomY) or (xmin, ymin, xmax, ymax)
        returns the area of the rectangle specified by r.
        """
        xmin, ymin, xmax, ymax = r
        return (xmax - xmin)*(ymax - ymin)

    def _rectangleOverlapArea(self, ra, rb):
        """
        computes the area of the overlap (i.e., intersection) b/w rectangle ra and 
        rectangle rb; both ra and rb are 4-tuples (xmin, ymin, xmax, ymax).
        """
        raxmin, raymin, raxmax, raymax = ra
        rbxmin, rbymin, rbxmax, rbymax = rb
        dx = min(raxmax, rbxmax) - max(raxmin, rbxmin)
        dy = min(raymax, rbymax) - max(raymin, rbymin)

        if dx >= 0 and dy >= 0:
            return dx*dy
        else:
            return -1

    def _processFrameForBeeMotion(self, frame, frame_copy, tick, roi_size, roi_dir, roi_filename,
                                  bckgrnd='MOG', lower_cnt_radius=3, upper_cnt_radius=15,
                                  overlap_thresh=OVERLAP_THRESH,
                                  draw_yb_roi_flag=False, draw_nb_roi_flag=False, save_yb_roi_flag=False,
                                  save_nb_roi_flag=False):
        # frame_sb = subtractBackground(frame.copy(), bckgrnd=bckgrnd)
        frame_sb = subtractBackground(frame_copy, bckgrnd=bckgrnd)

        return self._countBeeMotionROIsWithOverlap(frame_sb, frame, frame_copy, tick, roi_size, roi_dir,
                                                   roi_filename,
                                                   lower_cnt_radius=lower_cnt_radius,
                                                   upper_cnt_radius=upper_cnt_radius,
                                                   overlap_thresh=overlap_thresh,
                                                   draw_yb_roi_flag=draw_yb_roi_flag,
                                                   draw_nb_roi_flag=draw_nb_roi_flag,
                                                   save_yb_roi_flag=save_yb_roi_flag,
                                                   save_nb_roi_flag=save_nb_roi_flag)

        '''
        return self._processMotionContours(frame_sb, frame, frame_copy, tick, roi_size, roi_dir,
                                           roi_filename,
                                           lower_cnt_radius=lower_cnt_radius,
                                           upper_cnt_radius=upper_cnt_radius,
                                           draw_yb_roi_flag=draw_yb_roi_flag,
                                           draw_nb_roi_flag=draw_nb_roi_flag,
                                           save_yb_roi_flag=save_yb_roi_flag,
                                           save_nb_roi_flag=save_nb_roi_flag)
        '''

    # bmc = OmniVidBeeMotionCount()
    # total_count, dict  = bmc.countBeeMotionsInVid(vid_path)
    # total_count is the total number of bee motions detected in vid_path;
    # dict is a dictionary mappying frame numbers to bee motion counts in that frame.
    def countBeeMotionsInVid(self, vid_path,
                             show_frame_flag=True, save_frame_flag=False,
                             save_pfr_flag=False, save_roi_flag=False,
                             draw_yb_roi_flag=False, draw_nb_roi_flag=False,
                             save_yb_roi_flag=False, save_nb_roi_flag=False,
                             lower_cnt_radius=3,
                             upper_cnt_radius=15,
                             overlap_thresh=OVERLAP_THRESH,
                             frame_mod=1):
        #print('countBeeMotionsInVid: {}, draw_yb_roi_flag = {}, draw_nb_roi_flag = {}'.format(vid_path,
        #      draw_yb_roi_flag, draw_nb_roi_flag))
        vid_bee_motion_count = {}
        vid_bee_motion_count.clear()
        camera = cv2.VideoCapture(vid_path)
        FRAME_DIR, FRAME_FILENAME = self._createFrameDirAndFilename(vid_path, os.path.join(os.path.dirname(vid_path), 'frames/'))
        ROI_DIR, ROI_FILENAME = self._createRoiDirAndFilename(vid_path, os.path.join(os.path.dirname(vid_path), 'rois/'))
        PFR_DIR, PFR_FILENAME = self._createProcessedFrameDirAndFilename(vid_path, os.path.join(os.path.dirname(vid_path), 'pfr/'))
        FRAME_COUNTER = 0

        FRAME_DIR += CONVNET_NAME + '/'
        PFR_DIR   += CONVNET_NAME + '/'
        ROI_DIR   += CONVNET_NAME + '/'
                
        if FRAME_DIR[-1] != '/':
            raise Exception('countBeesInVid:` frame directory is not created properly...')
        if ROI_DIR[-1] != '/':
            raise Exception('countBeesInVid: roi directory is not created properly...')
        if PFR_DIR[-1] != '/':
            raise Exception('countBeesInVid: pfr directory is not created property...')

        print('FRAME_DIR = {}'.format(FRAME_DIR))
        print('FRAME_FILENAME = {}'.format(FRAME_FILENAME))
        print('ROI_DIR   = {}'.format(ROI_DIR))
        print('ROI_FILENAME = {}'.format(ROI_FILENAME))
        print('PFR_DIR   = {}'.format(PFR_DIR))
        print('PFR_FILENAME = {}'.format(PFR_FILENAME))

        # print 'FRAME_DIR=', FRAME_DIR
        # print 'FRAME_FILENAME=', FRAME_FILENAME

        # print FRAME_DIR
        # print FRAME_FILENAME

        # if not os.path.exists(FRAME_DIR):
        #     os.makedirs(FRAME_DIR)

        # if not os.path.exists(ROI_DIR):
        #     os.makedirs(ROI_DIR)

        # if not os.path.exists(PFR_DIR):
        #     os.makedirs(PFR_DIR)

        # print 'Processing video from Date:', videoCreationDate, ' and time:', videoCreationTime
        # keep looping

        while True:
            # grab the current frame
            (grabbed, frame) = camera.read()

            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if vid_path and not grabbed:
                print('Nothing grabbed; breaking...')
                break
            
            frame_copy = frame.copy()

            # save original frame if necessary.
            if save_frame_flag:
                orig_frame_path = FRAME_DIR + FRAME_FILENAME + '_' + str(FRAME_COUNTER) + '_orig.png'
                print('saving orig frame {}'.format(orig_frame_path))
                cv2.imwrite(orig_frame_path, frame)

            #print('FRAME SHAPE = {}'.format(frame.shape))
            #print('FRAME_COPY SHAPE {}'.format(frame_copy.shape))

            # if frame_mod == 10, then every 10th frame is used.
            if FRAME_COUNTER % frame_mod == 0:
                #print('FRAME_COUNTER % frame_mod == 0')
                pfr, frame_bee_motion_count = self._processFrameForBeeMotion(frame, frame_copy, 
                                                                             FRAME_COUNTER,
                                                                             ROI_SIZE,
                                                                             ROI_DIR,
                                                                             ROI_FILENAME,
                                                                             bckgrnd=BCKGRND,
                                                                             lower_cnt_radius=lower_cnt_radius,
                                                                             upper_cnt_radius=upper_cnt_radius,
                                                                             overlap_thresh=overlap_thresh,
                                                                             draw_yb_roi_flag=draw_yb_roi_flag,
                                                                             draw_nb_roi_flag=draw_nb_roi_flag,
                                                                             save_yb_roi_flag=save_yb_roi_flag,
                                                                             save_nb_roi_flag=save_nb_roi_flag)

                #print('show_frame_flag == {}'.format(show_frame_flag))
                #print('save_pfr_flag   == {}'.format(save_pfr_flag))

                # show processed frame pfr
                if show_frame_flag:
                    cv2.imshow('BeePi Bee Motion Detection', (cv2.resize(pfr, SHOW_FRAME_RESIZE)))
                    #cv2.imshow('Current Frame', pfr)

                if save_pfr_flag:
                    pfr_frame_path = PFR_DIR + PFR_FILENAME + '_' + str(FRAME_COUNTER) + '_pfr.png'
                    print('saving pfr {}'.format(pfr_frame_path))
                    cv2.imwrite(pfr_frame_path, pfr)
                
                vid_bee_motion_count[FRAME_COUNTER] = frame_bee_motion_count
                #print('vid_bee_motion_count[{}] = '.format(vid_bee_motion_count[FRAME_COUNTER]))
                #print(vid_bee_motion_count)
                
            FRAME_COUNTER += 1

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord('q'):
                break

        # print ('Total Bee count:', beeCountDictionary[beeCountKey])
        # print videoCounter + 1, '/', len(videos), ' done.'
        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
        
        total_count = 0
        for _, frame_count in vid_bee_motion_count.items():
            total_count += frame_count

        return total_count, vid_bee_motion_count


############################## UNIT TESTS ##########################################

# This VID_ROOT in for labgforce GPU computer
#VID_ROOT = 'PATH'
# VID_PATH03 = VID_ROOT + '4_5_1S_01.mp4'
# VID_PATH04 = VID_ROOT + '4_5_1S_02.mp4'
# VID_PATH05 = VID_ROOT + '4_5_1S_03.mp4'
# VID_PATH06 = VID_ROOT + '4_5_1S_04.mp4'
# VID_PATH07 = VID_ROOT + '4_5_1S_05.mp4'
# VID_PATH08 = VID_ROOT + '4_5_1S_06.mp4'
# VID_PATH09 = VID_ROOT + '4_7_1S_01.mp4'
# VID_PATH10 = VID_ROOT + '4_8_1S_01.mp4'

# ### LOW TRAFFIC
# VID_PATH11 = VID_ROOT + '4_8_1S_02_lt.mp4'

# ### MID TRAFFIC
# VID_PATH12 = VID_ROOT + '4_8_1S_03_mt.mp4'
# VID_PATH13 = VID_ROOT + '4_8_1S_04_mt.mp4'

# VID_PATH14 = VID_ROOT + '4_8_1S_05.mp4'
# VID_PATH15 = VID_ROOT + '4_10_1S_01.mp4'
# VID_PATH16 = VID_ROOT + '4_10_1S_02.mp4'

# ### HIGH TRAFFIC
# VID_PATH17 = VID_ROOT + '4_10_1S_03_ht.mp4'
# VID_PATH18 = VID_ROOT + '4_10_1S_04_ht.mp4'

# VID_PATH19 = VID_ROOT + '4_5_2S_01.mp4'
# VID_PATH20 = VID_ROOT + '4_5_2S_02.mp4'
# VID_PATH21 = VID_ROOT + '4_7_2S_01.mp4'
# VID_PATH22 = VID_ROOT + '4_8_2S_01.mp4'
# VID_PATH23 = VID_ROOT + '4_8_2S_02.mp4'
# VID_PATH24 = VID_ROOT + '4_10_2S_01.mp4'

# ### NO TRAFFIC
# VID_PATH25 = VID_ROOT + '4_5_1S_07_nt.mp4'

# ### HIGHT TRAFFIC
# VID_PATH26 = VID_ROOT + '4_7_2S_01_ht.mp4'
# VID_PATH27 = VID_ROOT + '4_5_2S_02_ht.mp4'

# ### LAASYA's SHADOW VIDS
# VID_PATH28 = VID_ROOT + 'laasya_shadow_vid1.mp4'
# VID_PATH29 = VID_ROOT + 'laasya_shadow_vid2.mp4'
# VID_PATH30 = VID_ROOT + 'laasya_shadow_vid3.mp4'

# VID_PATH31 = VID_ROOT + 'HT_vid02.mp4'
# VID_PATH32 = VID_ROOT + 'MT_vid02.mp4'
# VID_PATH33 = VID_ROOT + 'LT_vid02.mp4'
# VID_PATH34 = VID_ROOT + 'NT_vid02.mp4'

# VID_PATH35 = VID_DIR_R_4_5_may + '192_168_4_5-2019-05-10_15-34-52.mp4'
# VID_PATH36 = VID_DIR_R_4_5_may + '192_168_4_5-2019-05-10_16-04-52.mp4'
VID_PATH_TEST = 'valid/192_168_4_5-2018-07-08_14-40-10.mp4'
VID_PATH_VALID = 'valid/'

def vid_run_vgg16_1s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_1S
    tf.reset_default_graph()    

    print('restoring VGG16_1s model...')
    VGG16_1S = beepi_convnets.VGG16(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    VGG16_1S.load(PERSIST_BEE2_DIR + 'VGG16_1s' + '.tfl')
    print('VGG16_1s restored...')
    
    CONVNET_NAME = 'vgg16_1s'
    ROI_SIZE = ROI_SIZE_1S
    print('ROI_SIZE = {}'.format(ROI_SIZE))

    bmc = OmniVidBeeMotionCount(model=VGG16_1S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=False,
                                                        save_frame_flag=False,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=False,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)

    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
    return total_count, frame_dict

def vid_run_vgg16_2s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_2S
    tf.reset_default_graph()    

    print('restoring VGG16_2s model...')
    VGG16_2S = beepi_convnets.VGG16(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    VGG16_2S.load(PERSIST_BEE2_DIR + 'VGG16_2s' + '.tfl')
    print('VGG16_2s restored...')

    CONVNET_NAME = 'vgg16_2s'
    ROI_SIZE = ROI_SIZE_2S
    print('ROI_SIZE = {}'.format(ROI_SIZE))
    
    bmc = OmniVidBeeMotionCount(model=VGG16_2S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=False,
                                                        save_frame_flag=False,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=False,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)


    print(vid_path, total_count)
    for k, v in frame_dict.items():
        print('{} --> {}'.format(k, v))

    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
    return total_count

def vid_run_vgg16_2s_b(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_2S
    tf.reset_default_graph()    

    print('restoring VGG16_2s model...')
    VGG16_2S = beepi_convnets.VGG16(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    VGG16_2S.load(PERSIST_BEE2_DIR + 'VGG16_2s' + '.tfl')
    print('VGG16_2s restored...')

    CONVNET_NAME = 'vgg16_2s'
    ROI_SIZE = ROI_SIZE_2S
    print('ROI_SIZE = {}'.format(ROI_SIZE))

    bmc = OmniVidBeeMotionCount(model=VGG16_2S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=True,
                                                        save_frame_flag=True,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=True,
                                                        draw_nb_roi_flag=True,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)


    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))


def vid_run_convnetgs3_1s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_1S
    
    tf.reset_default_graph()    

    print('restoring ConvNetGS3_1s model...')
    ConvNetGS3_1S = beepi_convnets.ConvNetGS3(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ConvNetGS3_1S.load(PERSIST_BEE2_DIR + 'ConvNetGS3_1s' + '.tfl')
    print('ConvNetGS3_1s restored...')

    CONVNET_NAME = 'convnetgs3_1s'
    ROI_SIZE = ROI_SIZE_1S
    print('ROI_SIZE = {}'.format(ROI_SIZE))

    bmc = OmniVidBeeMotionCount(model=ConvNetGS3_1S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=False,
                                                        save_frame_flag=False,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=False,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)
    
    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
    return total_count, frame_dict

def vid_run_convnetgs3_2s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_2S
    
    tf.reset_default_graph()    

    print('restoring ConvNetGS3_2s model...')
    ConvNetGS3_2S = beepi_convnets.ConvNetGS3(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ConvNetGS3_2S.load(PERSIST_BEE2_DIR + 'ConvNetGS3_2s' + '.tfl')
    print('ConvNetGS3_2s restored...')

    CONVNET_NAME = 'convnetgs3_2s'
    ROI_SIZE = ROI_SIZE_2S
    print('ROI_SIZE = {}'.format(ROI_SIZE))

    bmc = OmniVidBeeMotionCount(model=ConvNetGS3_2S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=False,
                                                        save_frame_flag=False,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=False,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)
    
    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
    return total_count

def vid_run_convnetgs4_1s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_1S
    
    tf.reset_default_graph()    

    print('restoring ConvNetGS4_1s model...')
    ConvNetGS4_1S = beepi_convnets.ConvNetGS4(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ConvNetGS4_1S.load(PERSIST_BEE2_DIR + 'ConvNetGS4_1s' + '.tfl')
    print('ConvNetGS4_1s restored...')

    CONVNET_NAME = 'convnetgs4_1s'
    ROI_SIZE = ROI_SIZE_1S
    print('ROI_SIZE = {}'.format(ROI_SIZE))

    bmc = OmniVidBeeMotionCount(model=ConvNetGS4_1S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=False,
                                                        save_frame_flag=False,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=False,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)
    
    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
    return total_count, frame_dict

def vid_run_convnetgs4_2s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_2S
    
    tf.reset_default_graph()    

    print('restoring ConvNetGS4_1s model...')
    ConvNetGS4_2S = beepi_convnets.ConvNetGS4(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ConvNetGS4_2S.load(PERSIST_BEE2_DIR + 'ConvNetGS4_2s' + '.tfl')
    print('ConvNetGS4_2s restored...')

    CONVNET_NAME = 'convnetgs4_2s'
    ROI_SIZE = ROI_SIZE_2S
    print('ROI_SIZE = {}'.format(ROI_SIZE))
    
    bmc = OmniVidBeeMotionCount(model=ConvNetGS4_2S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=True,
                                                        save_frame_flag=True,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=True,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)
    
    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))


def vid_run_convnetgs5_2s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_2S
    
    tf.reset_default_graph()    

    print('restoring ConvNetGS5_2s model...')
    ConvNetGS5_2S = beepi_convnets.ConvNetGS5(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ConvNetGS5_2S.load(PERSIST_BEE2_DIR + 'ConvNetGS5_2s' + '.tfl')
    print('ConvNetGS5_2s restored...')

    CONVNET_NAME = 'convnetgs5_2s'
    ROI_SIZE = ROI_SIZE_2S
    print('ROI_SIZE = {}'.format(ROI_SIZE))
    
    bmc = OmniVidBeeMotionCount(model=ConvNetGS5_2S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=False,
                                                        save_frame_flag=False,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=False,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)
    
    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
    return total_count

def vid_run_resnet32_1s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_1S
    
    tf.reset_default_graph()    

    print('restoring ResNet32_1s model...')
    ResNet32_1S = beepi_convnets.ResNet32(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ResNet32_1S.load(PERSIST_BEE2_DIR + 'ResNet32_1s' + '.tfl')
    print('ResNet32_1s restored...')

    CONVNET_NAME = 'resnet32_1s'
    ROI_SIZE = ROI_SIZE_1S
    print('ROI_SIZE = {}'.format(ROI_SIZE))    

    bmc = OmniVidBeeMotionCount(model=ResNet32_1S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=False,
                                                        save_frame_flag=False,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=False,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)


    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
    return total_count, frame_dict


def vid_run_resnet32_2s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_2S
    
    tf.reset_default_graph()    

    print('restoring ResNet32_2s model...')
    ResNet32_2S = beepi_convnets.ResNet32(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ResNet32_2S.load(PERSIST_BEE2_DIR + 'ResNet32_2s' + '.tfl')
    print('ResNet32_2s restored...')

    CONVNET_NAME = 'resnet32_2s'
    ROI_SIZE = ROI_SIZE_2S
    print('ROI_SIZE = {}'.format(ROI_SIZE))
    
    bmc = OmniVidBeeMotionCount(model=ResNet32_2S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=True,
                                                        save_frame_flag=True,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=True,
                                                        draw_nb_roi_flag=True,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)


    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
            
            
def vid_run_convnet3_1s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_1S
    
    tf.reset_default_graph()    

    print('restoring ConvNet3_1s model...')
    ConvNet3_1S = beepi_convnets.ConvNet3(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ConvNet3_1S.load(PERSIST_BEE2_DIR + 'ConvNet3_1s' + '.tfl')
    print('ConvNet3_1s restored...')

    CONVNET_NAME = 'convnet3_1s'
    ROI_SIZE = ROI_SIZE_1S
    print('ROI_SIZE = {}'.format(ROI_SIZE))    
    
    bmc = OmniVidBeeMotionCount(model=ConvNet3_1S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=True,
                                                        save_frame_flag=True,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=True,
                                                        draw_nb_roi_flag=True,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)


    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))


def vid_run_convnet3_2s(vid_path, display_frame_counts_flag=False):
    global CONVNET_NAME
    global ROI_SIZE
    global ROI_SIZE_2S
    
    tf.reset_default_graph()    

    print('restoring ConvNet3_2s model...')
    ConvNet3_2S = beepi_convnets.ConvNet3(BEE2_IMWIDTH, BEE2_IMHEIGHT, BEE2_IMCHANNEL)
    ConvNet3_2S.load(PERSIST_BEE2_DIR + 'ConvNet3_2s' + '.tfl')
    print('ConvNet3_2s restored...')

    CONVNET_NAME = 'convnet3_2s'
    ROI_SIZE = ROI_SIZE_2S
    print('ROI_SIZE = {}'.format(ROI_SIZE))    

    bmc = OmniVidBeeMotionCount(model=ConvNet3_2S)
    total_count, frame_dict  = bmc.countBeeMotionsInVid(vid_path,
                                                        show_frame_flag=False,
                                                        save_frame_flag=False,
                                                        save_pfr_flag=False,
                                                        save_roi_flag=False,
                                                        draw_yb_roi_flag=False,
                                                        draw_nb_roi_flag=False,
                                                        save_yb_roi_flag=False,
                                                        save_nb_roi_flag=False,
                                                        lower_cnt_radius=LOWER_CNT_RADIUS,
                                                        upper_cnt_radius=UPPER_CNT_RADIUS,
                                                        overlap_thresh=OVERLAP_THRESH)


    print(total_count)
    if display_frame_counts_flag:
        for k, v in frame_dict.items():
            print('{} --> {}'.format(k, v))
    return total_count
            
            
def unit_test_02():
    yb01 = 'PATH_TO_FILE/yb.png'
    yb02 = 'PATH_TO_FILE/yb2.png'
    yb03 = 'PATH_TO_FILE/yb3.png'
    yb04 = 'PATH_TO_FILE/yb4.png'
    nb01 = 'PATH_TO_FILE/nb.png'
    nb02 = 'PATH_TO_FILE/nb2.png'

    bmc = OmniVidBeeMotionCount(model=ConvNetGS3_1S)
    y = bmc.containsBee(cv2.imread(yb01))
    print('y={}'.format(y))
    y = bmc.containsBee(cv2.imread(yb02))
    print('y={}'.format(y))
    y = bmc.containsBee(cv2.imread(yb03))
    print('y={}'.format(y))
    y = bmc.containsBee(cv2.imread(yb04))
    print('y={}'.format(y))
    y = bmc.containsBee(cv2.imread(nb01))
    print('y={}'.format(y))
    y = bmc.containsBee(cv2.imread(nb02))
    print('y={}'.format(y))

import glob
import os
import re

def get_frame_number(fp):
    bn = os.path.basename(fp)
    pos = [m.start() for m in re.finditer('_', bn)]
    return int(bn[pos[-2]+1:pos[-1]]) 
    
def unit_test_03():
    fp = glob.glob('research/EBM/bee_vid_detection/video/pfr/4_5-05-06-18_17:45_pfr/*.png')
    fp.sort(key=get_frame_number)
    for p in fp:
        print(p)

def unit_test_04():
    pfr_path01 = 'PATH_TO_FILE/28may19/video/pfr/05-15-18_13:00_pfr/*.png'
    pfr_path02 = 'PATH_TO_FILE/28may19/video/pfr/4_5-05-06-18_17:45_pfr/*.png'
    fp = glob.glob(pfr_path02)
    fp.sort(key=get_frame_number)
    for p in fp:
        print(p)
        img = cv2.imread(p)
        w, h, _ = img.shape
        img = cv2.resize(img, (1600, 760))
        cv2.imshow('Current Frame', img)
        print(img.shape)
        del img
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def unit_test_05():
    fp = '/video/frames/4_8_1S_03_mt_orig/vgg16_1s/4_8_1S_03_mt741_orig.png'
    img = cv2.imread(fp)
    t = {}
    t[0] = [(1394, 687, 1544, 837)]
    t[1] = [(1770, 685, 1920, 835), (1770, 684, 1920, 834), (1770, 672, 1920, 822), (1769, 653, 1919, 803)]
    t[2] = [(1725, 658, 1875, 808)]
    t[3] = [(1366, 647, 1516, 797), (1376, 640, 1526, 790)]
    t[4] = [(1636, 635, 1786, 785)]
    t[5] = [(1326, 126, 1476, 276)]
    t[6] = [(1505, 122, 1655, 272), (1518, 118, 1668, 268), (1526, 113, 1676, 263), (1541, 111, 1691, 261),
            (1538, 106, 1688, 256), (1553, 106, 1703, 256), (1565, 101, 1715, 251), (1563, 95, 1713, 245),
            (1578, 86, 1728, 236)]
    t[7] = [(405, 88, 555, 238), (404, 68, 554, 218)]
    t[8] = [(803, 0, 953, 150), (778, 0, 928, 150), (826, 0, 976, 150), (796, 0, 946, 150), (785, 0, 935, 150)]
    t[9] = [(410, 0, 560, 150), (405, 0, 555, 150)]
    for i, c in t.items():
        x1, y1, x2, y2 = c[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), DRAW_YB_ROI_COLOR, 2)
    cv2.imshow('Clusters', img)
    cv2.waitKey()

def unit_test_06():
    fp = '/video/frames/4_8_1S_03_mt_orig/vgg16_1s/4_8_1S_03_mt742_orig.png'
    img = cv2.imread(fp)
    t = {}
    t[0] = [(1344, 686, 1494, 836)]
    t[1] = [(1636, 683, 1786, 833), (1618, 690, 1768, 840)]
    t[2] = [(1770, 682, 1920, 832), (1770, 684, 1920, 834)]
    t[3] = [(1677, 682, 1827, 832), (1660, 685, 1810, 835)]
    t[4] = [(1370, 659, 1520, 809), (1366, 647, 1516, 797), (1386, 639, 1536, 789)]
    t[5] = [(1725, 658, 1875, 808), (1717, 639, 1867, 789)]
    t[6] = [(1633, 636, 1783, 786)]
    t[7] = [(1077, 304, 1227, 454)]
    t[8] = [(1504, 130, 1654, 280), (1519, 123, 1669, 273), (1526, 119, 1676, 269), (1541, 117, 1691, 267),
            (1554, 110, 1704, 260), (1549, 107, 1699, 257), (1576, 104, 1726, 254), (1585, 89, 1735, 239)]
    t[9] = [(1326, 126, 1476, 276)]
    t[10] = [(405, 88, 555, 238), (404, 68, 554, 218)]
    t[11] = [(884, 0, 1034, 150)]
    t[12] = [(410, 0, 560, 150), (405, 0, 555, 150)]
    for i, c in t.items():
        x1, y1, x2, y2 = c[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), DRAW_YB_ROI_COLOR, 2)
    cv2.imshow('Clusters', img)
    cv2.waitKey()

def rectOverlapArea(ra, rb):
    """
    computes the area of the overlap (i.e., intersection) b/w rectangle ra and 
    rectangle rb; both ra and rb are 4-tuples (xmin, ymin, xmax, ymax).
    """
    raxmin, raymin, raxmax, raymax = ra
    rbxmin, rbymin, rbxmax, rbymax = rb
    dx = min(raxmax, rbxmax) - max(raxmin, rbxmin)
    dy = min(raymax, rbymax) - max(raymin, rbymin)

    if dx >= 0 and dy >= 0:
        return dx*dy
    else:
        return -1

def unit_test_07():
    ROI_AREA = ROI_SIZE_1S
    sim_pred = lambda bmr1, bmr2: rectOverlapArea(bmr1, bmr2)/ROI_AREA
    cls = clusters()
    bmrs = [(1344, 686, 1494, 836), (1636, 683, 1786, 833), (1618, 690, 1768, 840),
            (1770, 682, 1920, 832), (1770, 684, 1920, 834), (1677, 682, 1827, 832), (1660, 685, 1810, 835),
            (1370, 659, 1520, 809), (1366, 647, 1516, 797), (1386, 639, 1536, 789), (1725, 658, 1875, 808),
            (1717, 639, 1867, 789), (1633, 636, 1783, 786), (1077, 304, 1227, 454), (1504, 130, 1654, 280),
            (1519, 123, 1669, 273), (1526, 119, 1676, 269), (1541, 117, 1691, 267), (1554, 110, 1704, 260),
            (1549, 107, 1699, 257), (1576, 104, 1726, 254), (1585, 89, 1735, 239), (1326, 126, 1476, 276),
            (405, 88, 555, 238), (404, 68, 554, 218), (884, 0, 1034, 150), (410, 0, 560, 150),
            (405, 0, 555, 150)]
    for bmr in bmrs:
        cls.cluster(bmr, sim_pred, OVERLAP_THRESH)
    for i, cl in enumerate(cls.getClusters()):
        print('{}) {}'.format(i, cl))
    


def createBeeCountCSV(VIDEO_NAME, DICT_DATA):
    CSV_COLUMNS = ['FRAME', 'COUNT']
    import csv
    #writing
    with open(VIDEO_NAME + '_beeMotion.csv', 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(CSV_COLUMNS)
        for k,v in DICT_DATA.items():
            writer.writerow([k,v])

# When given a video directory, it returns a bee count dictionary. The video names need to be in the format %m-%d-%y_%H:%M
# 1S: VID_PATH_06, VID_PATH08, VID_PATH14, VID_PATH18,
# 2S: VID_PATH19 w/ convnet3_2s, VID_PATH20 w/ vgg16_2s, VID_PATH21 w/ vgg16_2s, VID_PATH22 w/ vgg16_2s, convnetgs3_2s
# make 2S videos out of VID_PATH19
# VID_PATH12 is a great one for false positives, because of a different background
# VID_PATH15 is a great one for false positives against a different background.
if __name__=='__main__':
    #unit_test_07()    
    #vid_run_convnetgs3_1s(VID_PATH06)
    #vid_run_vgg16_2s_b(VID_PATH19, display_frame_counts_flag=True)
    
    ### NO TRAFFIC; MOVIES
    #vid_run_vgg16_1s(VID_PATH25)       ## frame_mod = 1; bmc = 151; human count = 73
    #vid_run_resnet32_1s(VID_PATH25)    ## frame_mod = 1; bmc =  75; human count = 73
    #vid_run_convnetgs3_1s(VID_PATH25)  ## frame_mod = 1; bmc = 182; human count = 73
    #vid_run_convnetgs4_1s(VID_PATH25)  ## frame_mod = 1; bmc = 127; human count = 73
    
    ### LOW TRAFFIC -- MOVIES
    #vid_run_vgg16_1s(VID_PATH11)       ## frame_mod=1; bmc = 47; human count = 353
    #vid_run_resnet32_1s(VID_PATH11)    ## frame_mod=1; bmc = 25; human count = 353
    #vid_run_convnetgs3_1s(VID_PATH11)  ## frame_mod=1; bmc = 57; human count = 353
    #vid_run_convnetgs4_1s(VID_PATH11)  ## frame_mod=1; bmc = 43; human count = 353

    ### MID TRAFFIC
    #vid_run_vgg16_1s(VID_PATH12)       ## MT with FRAME_MOD=30; bmc = 688
    #vid_run_vgg16_1s(VID_PATH13)      ## MT with FRAME_MOD=30; bmc = 980
    #vid_run_convnetgs4_1s(VID_PATH12) ## MT with FRAME_MOD=30; bmc = 651
    #vid_run_convnetgs4_1s(VID_PATH13) ## MT with FRAME_MOD=30; bmc = 525
    #vid_run_resnet32_1s(VID_PATH12)   ## MT with frame_mod=30; bmc = 594
    #vid_run_resnet32_1s(VID_PATH13)   ## MT with frame_mod=30; bmc = 587

    # MID TRAFFIC; frame_mod = 1; MOVIES; MID is high traffic for now;
    #vid_run_vgg16_1s(VID_PATH12)        ## frame_mod=1; bmc = 16647; human count = 5738;
    #vid_run_resnet32_1s(VID_PATH12)     ## frame_mod=1; bmc = 13362; human count = 5738;
    #vid_run_convnetgs3_1s(VID_PATH12)   ## frame_mod=1; bmc = 16569; human count = 5738;
    #vid_run_convnetgs4_1s(VID_PATH12)   ## frame_mod=1; bmc = 15109; human count = 5738;

    ### HIGH TRAFFIC; This mid traffic for now
    #vid_run_vgg16_1s(VID_PATH17) ## HT with frame_mod=30; bmc = 909
    #vid_run_vgg16_1s(VID_PATH18) ## HT with frame_mod=30; bmc = 774
    #vid_run_convnetgs4_1s(VID_PATH17) ## HT with FRAME_MOD=30; bmc = 810
    #vid_run_convnetgs4_1s(VID_PATH18) ## HT with frame_mod=30; bmc = 467; false positives
    #vid_run_resnet32_1s(VID_PATH17) ## MT with frame_mod=30; bmc = 602
    #vid_run_resnet32_1s(VID_PATH18) ## MT with frame_mod=30; bmc = 129

    ## FRAME_MOD experiments
    #vid_run_vgg16_2s(VID_PATH22) # frame_mod=1;  bmc = 8587
    #vid_run_vgg16_2s(VID_PATH22) # frame_mod=5;  bmc = 1989
    #vid_run_vgg16_2s(VID_PATH22) # frame_mod=10; bmc = 1038
    #vid_run_vgg16_2s(VID_PATH22) # frame_mod=15; bmc = 638
    #vid_run_vgg16_2s(VID_PATH22) # frame_mod=20; bmc = 499
    #vid_run_vgg16_2s(VID_PATH22) # frame_mod=25; bmc = 393
    #vid_run_vgg16_2s(VID_PATH22) # frame_mod=30; bmc = 322

    #vid_run_vgg16_2s(VID_PATH26)      ## HT with frame_mod=1;  bmc = 6655
    # MOVIES; HIGHT TRAFFIC; mid traffic now
    #vid_run_vgg16_2s(VID_PATH27)      ## HT with frame_mod=1;   bmc = 6638;  human count = 2924
    #vid_run_resnet32_2s(VID_PATH27)   ## HT with frame_mod=1;  bmc = 145;   human count = 2924
    #vid_run_convnetgs3_1s(VID_PATH27) ## HT with frame_mod=1;  bmc = 597;   human count = 2924
    #vid_run_convnetgs4_1s(VID_PATH27) ## HT with frame_mod=1;  bmc = 316;   human count = 2924

    #HT_vid02.mp4
    #vid_run_vgg16_1s(VID_PATH31)      ## frame_mod=1; bmc = 13,002; human count = unavail.
    #vid_run_resnet32_1s(VID_PATH31)   ## frame_mod=1; bmc = 9,716;  human count = unavail.
    #vid_run_convnetgs3_1s(VID_PATH31) ## frame_mod=1; bmc = 9,960; human count = unavail.
    #vid_run_convnetgs4_1s(VID_PATH31) ## frame_mod=1; bmc = 9,885; human count = unavail.

    #MT_vid02.mp4
    #vid_run_vgg16_1s(VID_PATH32)   ## frame_mod=1; bmc = 1,095; human count = unavail.
    #vid_run_resnet32_1s(VID_PATH32) ## frame_mod=1; bmc = 343; human count = unavail.
    #vid_run_convnetgs3_1s(VID_PATH32) ## frame_mod=1; bmc = 1,179; human count = unavail.
    #vid_run_convnetgs4_1s(VID_PATH32) ## frame_mod=1; bmc = 871; human count = unavail.

    #LT_vid02.mp4
    #vid_run_vgg16_2s(VID_PATH33)      ## frame_mod=1; bmc = 530; human count = unavail.
    #vid_run_resnet32_1s(VID_PATH33)   ## frame_mod=1; bmc = 1,101; human count = unavail.
    #vid_run_convnetgs3_1s(VID_PATH33) ## frame_mod=1; bmc = 1,439; human count = unavail.
    #vid_run_convnetgs4_1s(VID_PATH33) ## frame_mod=1; bmc = 10,000; human count = unavail.

    #NT_vid02.mp4
    #vid_run_vgg16_2s(VID_PATH34)      ## frame_mod=1; bmc = 140; human count = unavail.
    #vid_run_resnet32_1s(VID_PATH34)   ## frame_mod=1;  bmc = 188; human count = unavail.
    #vid_run_convnetgs3_1s(VID_PATH34) ## frame_mod=1; bmc = 175; human count = unavail.
    #vid_run_convnetgs4_1s(VID_PATH34)  ## frame_mod=1; bmc = 173; human count = unavail.

    #HT_vid03.mp4

    ## FRAME_MOD experiments
    #vid_run_vgg16_1s(VID_PATH08) # 
    
    ## bee motion count experiments
    #vid_run_vgg16_1s(VID_PATH11)      ## LT with FRAME_MOD=1, bmc = 47
    #vid_run_resnet32_1s(VID_PATH11)   ## LT with FRAME_MOD=1, bmc = 25
    #vid_run_convnetgs3_1s(VID_PATH11) ## LT with FRAME_MOD=1, bmc = 57
    #vid_run_convnetgs4_1s(VID_PATH11) ## LT with FRAME_MOD=1, bmc = 43

    #vid_run_vgg16_1s(VID_PATH12)      ## frame_mod=1, bmc = 16,647
    #vid_run_vgg16_1s(VID_PATH12)      ## frame_mod = 5; bmc = 4246;
    #vid_run_vgg16_1s(VID_PATH12)      ## frame_mod = 3; bmc = 6938;
    #vid_run_resnet32_1s(VID_PATH12)   ## FRAME_MOD=25, bmc = 674    
    #vid_run_convnetgs3_1s(VID_PATH12) ## FRAME_MOD=25, bmc = 809
    #vid_run_convnetgs4_1s(VID_PATH12) ## FRAME_MOD=25, bmc = 738
    
    #vid_run_resnet32_1s(VID_PATH02)
    #vid_run_convnet3_1s(VID_PATH02)
    #vid_run_convnetgs4_1s(VID_PATH18)
    #vid_run_resnet32_1s(VID_PATH12)
    #vid_run_vgg16_2s(VID_PATH22)
    #vid_run_convnetgs3_2s(VID_PATH19)
    #vid_run_convnetgs4_2s(VID_PATH19)
    #vid_run_resnet32_2s(VID_PATH19)
    #vid_run_convnet3_2s(VID_PATH19)
    #vid_run_vgg16_2s(VID_PATH23)
    #vid_run_convnetgs3_2s(VID_PATH19)
    #vid_run_convnetgs4_2s(VID_PATH21)
    #vid_run_resnet32_2s(VID_PATH21)
    #vid_run_convnetgs4_2s(VID_PATH22)
    #vid_run_resnet32_2s(VID_PATH19)
    #vid_run_convnet3_2s(VID_PATH19)

    # LAASYA's SHADOW EXPERIMENTS; Aug 1, 2019; she used it for
    # her defense on Aug 5, 2019.
    #vid_run_vgg16_1s(VID_PATH28)
    #vid_run_vgg16_1s(VID_PATH29)
    #vid_run_vgg16_1s(VID_PATH30)
    
    # vid_run_vgg16_1s(VID_PATH36)
    # vid_run_vgg16_2s(VID_PATH_TEST)


    
   	#######################################################################for piv and bee motions on all videos##########################################
    # import sys
    # if sys.argv[1].endswith('.mp4'):
    #     VID_PATH = 'video/'+sys.argv[1]
    #     print(VID_PATH)
    #     count = vid_run_convnetgs5_2s(VID_PATH)
    #     print(count)
    #     import csv
    #     with open('video/beeCount_convnet5_2s.csv', 'a') as output_file:
    #         writer = csv.writer(output_file)
    #         writer.writerow([sys.argv[1], count])
    ######################################################################################################################################################

    #######################################################################for piv and bee motions calculate on each frame##########################################
    import sys
    if sys.argv[1].endswith('.mp4'):
        VID_PATH = 'applied_science_video/'+sys.argv[1]
        print(VID_PATH)
        count, frame_dict_counts = vid_run_convnetgs4_1s(VID_PATH)
        createBeeCountCSV(VID_PATH, frame_dict_counts)
    ######################################################################################################################################################

pass