# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:44:07 2017

@author: zck
"""

import cv2
import videoCapture
import webcamCapture
import faceDetector
import numpy as np
import argparse
import time

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some parameters.')
    
    parser.add_argument('--webcam', action='store_true', default=True)
    
    args = parser.parse_args()
    
    if args.webcam:
        
        cap = webcamCapture.Capture()
        
    else:
        
        cap = videoCapture.Capture('testdata/hap001.avi')
    
    mtcnn = faceDetector.MTCNN(minsize=50)
    
    ret, frame = cap.get_frame()
    
    while ret:
        
        img_rgb = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2RGB)
        
        tim1 = time.time()
        
        bounding_boxes, landmarks = mtcnn.detect_face(img_rgb)
        
        new_frame = mtcnn.draw_bounding_box(frame, bounding_boxes)
        
        tim2 = time.time()
        
        print('FPS = {}'.format( 1.0/(tim2-tim1) ))
        
        cv2.imshow('1', frame)
        
        cv2.imshow('2', new_frame)
        
        key = cv2.waitKey(50)
        
        if 27==key:
            
            break
                        
        ret, frame = cap.get_frame()
        
#        break
    
    cv2.destroyAllWindows()
    
    del cap
    
    