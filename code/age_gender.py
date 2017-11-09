# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:51:34 2017

@author: zck
"""

import sys
sys.path.append('caffe/python')
import caffe
import numpy as np
import cv2

class Basic:
    
    def __init__(self, model_def, model_weights, gpu_id=0):
        
        caffe.set_device(gpu_id)
        
        caffe.set_mode_gpu()
        
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)    
    
    def predict(self, images, bottom='data', top='prob'):
        
        ret = []
        
        net = self.net
        
        for image in np.array(images, dtype=np.float32):
            
            img = cv2.resize(image, (224,224))
            
            img[:, :, 0] -= 123.68
            
            img[:, :, 1] -= 116.779
            
            img[:, :, 2] -= 103.939
            
            img = img.transpose((2, 0, 1))
            
            net.blobs[bottom].data[...] = np.array([img])

            output = net.forward()
            
            prob = net.blobs[top].data[0]
            
            ret.append(prob)
        
        return np.array(ret)


class Age(Basic):
        
    def __init__(self, model_def, model_weights, gpu_id=0):
        
        super().__init__(model_def, model_weights, gpu_id)
        
    def predict(self, images, bottom='data', top='prob'):
        
        age_probs = super().predict(images, bottom, top)
        
        self.age_probs = age_probs
        
        ret = []
        
        for age_prob in age_probs:
            
            e = 0.0
            
            for i in range(len(age_prob)):
                
                e += i*age_probs[0][i]
            
            ret.append(e)
        
        return ret
    

class Gender(Basic):

    def __init__(self, model_def, model_weights, gpu_id=0):
        
        super().__init__(model_def, model_weights, gpu_id)
        
    def predict(self, images, bottom='data', top='prob'):
        
        gender_probs = super().predict(images, bottom, top)
    
        ret = []
        
        for prob in gender_probs:
            
            if prob[0] > prob[1]:
                
                ret.append('Female')
            
            else:
                
                ret.append('Male')
        
        return ret
        
    
if __name__ == '__main__':
    
    img0 = cv2.imread('testdata/81800_1986-06-13_2011.jpg')
    
    """Gender: 0 for female and 1 for male, NaN if unknown
    """
    gender = Gender('weights/gender.prototxt', 'weights/gender.caffemodel')
    
    gender_probs = gender.predict([img0])
    
    print(gender_probs)
    
    
    age = Age('weights/age.prototxt', 'weights/dex_chalearn_iccv2015.caffemodel')
    
    age_probs = age.predict([img0])
    
    e = age_probs[0]
        
    print(e)
    
    
    
    
    