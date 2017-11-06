# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:44:07 2017

@author: zck
"""

import cv2
import videoCapture
import faceDetector
import numpy as np

if __name__ == '__main__':
    
    cap = videoCapture.Capture('testdata/hap001.avi')
    
    mtcnn = faceDetector.MTCNN()
    
    ret, frame = cap.get_frame()
    
    while ret:
        
        img_rgb = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2RGB)
        
        bounding_boxes, landmarks = mtcnn.detect_face(img_rgb)
        
        new_frame = mtcnn.draw_bounding_box(frame, bounding_boxes)
        
        cv2.imshow('1', frame)
        
        cv2.waitKey(1)
        
        cv2.imshow('2', new_frame)
        
        cv2.waitKey(-1)
                        
        ret, frame = cap.get_frame()
        
        break
    
    cv2.destroyAllWindows()
    
    
    