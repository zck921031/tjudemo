# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 14:44:07 2017

@author: zck
"""

import cv2

class Capture():
    
    def __init__(self):
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():

            print('Failed to load camera.')
            
            raise(Exception('Failed to load camera.'))
            
        self.cap = cap
        
        
    def get_frame(self):
        
        ret, frame = self.cap.read()
        
        return ret, frame
        


if __name__ == '__main__':
    
    cap = Capture()
    
    frame = cap.get_frame()
    