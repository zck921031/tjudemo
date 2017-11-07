# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:44:07 2017

@author: zck
"""

import cv2

class Capture():
    
    def __init__(self, width=1280, height=960):
        
        cap = cv2.VideoCapture(0)        
    
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
        if not cap.isOpened():

            print('Failed to load camera.')
            
            raise(Exception('Failed to load camera.'))
            
        self.cap = cap
          
    def get_frame(self):
        
        ret, frame = self.cap.read()
        
        return ret, frame
        

if __name__ == '__main__':
    
    cap = Capture()
    
    ret, frame = cap.get_frame()    
    
    while ret:
        
        ret, frame = cap.get_frame()
        
        cv2.imshow('webcam', frame)
        
        key = cv2.waitKey(1)
        
        print(key)
        
        if 27==key:
            
            break
        
        ret, frame = cap.get_frame()
    
    cv2.destroyAllWindows()
    
    cap.cap.release()
    
    