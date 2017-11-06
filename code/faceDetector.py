# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:05:15 2017

@author: zck

img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
img_size = np.asarray(img.shape)[0:2]
bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
"""

from align import detect_face
import tensorflow as tf
import numpy as np
import cv2

class MTCNN():
    """Face detection and alignment using Multi-Task CNN.
    
    """
    def __init__(self, minsize=20, factor=0.709, threshold=[0.8, 0.9, 0.9]):
        
        with tf.device('/gpu:0'):
            
            with tf.Graph().as_default():
                
                gpu_options = tf.GPUOptions(allow_growth=True)
                
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                
                with sess.as_default():
                    
                    pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align/')
                    
        self.pnet = pnet
        
        self.rnet = rnet
        
        self.onet = onet
        
        self.minsize = minsize
        
        self.factor = factor
        
        self.threshold = threshold        
        
        
    def detect_face(self, img_rgb):
        """
        
        Input: R-G-B image matrix.
        
        Output: bounding_boxes (N x 5), landmarks (10 x N)
        
            bounding_box: [left, up, right, down, score]
        """
        
        bounding_boxes, landmarks = detect_face.detect_face(img_rgb, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        
        return bounding_boxes, landmarks
    
    
    def draw_bounding_box(self, img, bounding_boxes, margin_percent=0.1, landmarks=None, color=(0,0,255), debug=False):
        
        video_height, video_width = img.shape[:2]
        
        frame = np.copy(img)
        
        for bb in np.array(bounding_boxes, np.int32):
        
            weight = bb[2] - bb[0]
            
            height = bb[3] - bb[1]
            
            left  = int(bb[0] - margin_percent * weight)
            
            up    = int(bb[1] - margin_percent * height)
            
            right = int(bb[2] + margin_percent * weight)
            
            down  = int(bb[3] + margin_percent * height)
            
            left  = int(max(left, 0))
            
            up    = int(max(up, 0))
            
            right = int(min(right, video_width))
            
            down  = int(min(down, video_height))       
            
            cv2.rectangle(frame, (left, up), (right, down), color, 2)
            
            if debug:
                
                crop_face = img[up:down, left:right, :]
                
                plt.imshow(np.uint8(crop_face))
            
                plt.show()
        
        return frame
    
if __name__ == '__main__':
    
    from scipy import misc
    import matplotlib.pyplot as plt
    
    mtcnn = MTCNN()
    
    img = misc.imread('testdata/image01.jpg', mode='RGB')
    
#    misc.imshow(img)
    
    plt.imshow(np.uint8(img))
    
    plt.show()
    
    video_height, video_width = img.shape[:2]
    
    bounding_boxes, landmarks = mtcnn.detect_face(img)
    
    margin_percent = 0.1
    
    frame = mtcnn.draw_bounding_box(img, bounding_boxes, color=(255,0,0), debug=True)
        
    plt.imshow(np.uint8(frame))
        
    plt.show()
    
    misc.imsave('testdata/image01-face-detection.jpg', frame)
        
        