# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:44:07 2017

@author: zck
"""

import cv2

class Capture():
    
    def __init__(self, videoname):
        
        cap = cv2.VideoCapture(videoname)
            
        if not cap.isOpened():

            print('Failed to load video.')
            
            raise(Exception('Failed to load video.'))
            
        self.cap = cap
        
        
    def get_frame(self):
        
        ret, frame = self.cap.read()
        
        return ret, frame
        

if __name__ == '__main__':
    
    cap = Capture('testdata/hap001.avi')
    
    ret, frame = cap.get_frame()
    
    while ret:
        
        cv2.imshow('1', frame)
        
        cv2.waitKey(100)
        
        ret, frame = cap.get_frame()
    
    cv2.destroyAllWindows()
    
    
    