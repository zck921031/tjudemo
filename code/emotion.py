# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 14:00:44 2017

@author: zck
"""

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from convnetskeras.convnets import AlexNet
from scipy.misc import imread, imresize
from skimage import color
from sklearn.neighbors import NearestNeighbors
from keras.layers import Activation


class EmotionPredictor:
    
    def __init__(self):
        
        model_img = AlexNet(heatmap=False)
        
        inp = model_img.input
        
        dense_1 = model_img.get_layer('flatten').output
        
        feats = Dense(32, activation='linear')(dense_1)
        
        model = Model(input=inp, output=feats)
        
#        model.compile(loss='mse', optimizer=RMSprop(), metrics=[])
        
        model.load_weights('weights/jaffe.0.012.frcnn32.0.keras.h5')
        
        self.model1 = model
                
        inp = Input(shape=(32,))
        
        feats = Dense(32, activation='relu', name='feats')(inp)
        
        fc1 = Dense(16, activation='relu')(feats)
        
        fc2 = Dense(6, activation='linear')(fc1)
        
        label = Activation('softmax', name='label')(fc2)
        
        label_predictor = Model(input=inp, output=label)
        
        feature_predictor = Model(input=[inp], output=[feats])

        label_predictor.load_weights('weights/label_predictor.weights')
        
        self.feature_predictor = feature_predictor
        
        self.label_predictor = label_predictor
        
        self.yTr = np.load('weights/yTr.npy')
    
        self.featuresTr = np.load('weights/featuresTr.npy')
        
        self.fTr = np.load('weights/fTr.npy')
        
        self.yFuzzy = np.load('weights/yFuzzy.npy')
        
    def predict_softmax(self, imgs):

        return self.label_predictor.predict(self.model1.predict(imgs))
        
    def predict_knn(self, imgs, k=3):
        
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(self.fTr)
        
        fTe = self.feature_predictor.predict(self.model1.predict(imgs))
        
        distances, indices = nbrs.kneighbors(fTe)
        
        ret = np.array([np.mean(self.yFuzzy[ind], axis=0) for ind in indices])

        ret = ret/5.0
        
#        ret = (ret-1.0)/4.0
        
        return ret
    
    def preprocess(self, imgs, img_size=None, crop_size=None, usegray=True, out=None):
        """color_mode="rgb"
        
        """
        
        if len(imgs)==0:
            
            return np.array([])
    
        img_list = []
    
        for img in imgs:
            
            if img_size:
                
                img = imresize(img,img_size)
    
            img = img.astype('float32')
            
            if usegray:
                
                img = color.gray2rgb(color.rgb2gray(img/255.0))*255.0            
                
            # We normalize the colors (in RGB space) with the empirical means on the training set
            img[:, :, 0] -= 123.68
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 103.939
            
            img = img.transpose((2, 0, 1))
    
            if crop_size:
                
                img = img[:,(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                          ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]
    
            img_list.append(img)
    
        try:
            
            img_batch = np.stack(img_list, axis=0)
            
            return img_batch
        
        except:
            
            raise ValueError('when img_size and crop_size are None, images'
                    ' in image_paths must have the same shapes.')
    
        

if __name__ == '__main__':
    
    ep = EmotionPredictor()
    
    img0 = imread('testdata/0-face.jpg', mode='RGB')
    
    imgs = [img0]
    
    imagesTe = ep.preprocess(imgs, img_size=(256,256), crop_size=(227,227))
    
    yy = ep.predict_knn(imagesTe)
    
    