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
import emotion
import matplotlib.pyplot as plt
import age_gender

def draw_emotion_radar(data):
            #标签
    labels = np.array(['HAP', 'SAD', 'SUR', 'ANG', 'DIS', 'FEA'])
    
#    data = (data-1.0)/4.0
    
    #数据个数
    dataLenth = 6
    
    #========自己设置结束============
    
    angles = np.linspace(0, 2*np.pi, dataLenth, endpoint=False)
    data = np.concatenate((data, [data[0]])) # 闭合
    angles = np.concatenate((angles, [angles[0]])) # 闭合
    
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)# polar参数
    ax.plot(angles, data, 'bo-', linewidth=2)# 画线
    ax.fill(angles, data, facecolor='r', alpha=0.25)# 填充
    ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="Song", fontsize=12)
#        ax.set_title("matplotlib", va='bottom', fontproperties="SimHei")
    ax.set_rlim(0, 1)
    ax.grid(True)
    
    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
    plt.close()
    
    return img
    
#    plt.show()

class Screen():
    
    def __init__(self, basic):
        
        self.img = np.zeros((basic.shape[0], basic.shape[1], basic.shape[2]))
        
        self.img[:basic.shape[0], :basic.shape[1], :basic.shape[2]] = basic
        
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some parameters.')
    
    parser.add_argument('--offline', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if not args.offline:
        
        cap = webcamCapture.Capture(width=800, height=600)
        
    else:
        
        cap = videoCapture.Capture('testdata/hap01.avi')
    
    """Several Networks used in this project.    
    """
    mtcnn = faceDetector.MTCNN(minsize=50)
    
    emotion_predictor = emotion.EmotionPredictor()
    
    agenet = age_gender.Age('weights/age.prototxt', 'weights/dex_chalearn_iccv2015.caffemodel')
    
    gendernet = age_gender.Gender('weights/gender.prototxt', 'weights/gender.caffemodel')
    
    ret, frame = cap.get_frame()
    
    while ret:
        
        img_rgb = cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2RGB)
        
        tim1 = time.time()
        
        bounding_boxes, landmarks = mtcnn.detect_face(img_rgb)
        
        new_frame = mtcnn.draw_bounding_box(frame, bounding_boxes)
        
        screen = Screen(frame)
        
        if len(bounding_boxes)>0:
        
            faces = mtcnn.crop_faces(frame, bounding_boxes, margin_percent=0.3)        
            
            faces_rgb = mtcnn.crop_faces(img_rgb, bounding_boxes, margin_percent=0.3)
            
            faces_norm = emotion_predictor.preprocess(faces_rgb, img_size=(256,256), crop_size=(227,227))
            
#            emotions = emotion_predictor.predict_softmax(faces_norm)
            
            emotions = emotion_predictor.predict_knn(faces_norm)
            
            radar = draw_emotion_radar(emotions[0])
            
            cv2.imshow("Emotion Radar", radar)
            
            print(emotions)
            
            """Age + Gender
            """
            for (face, bb) in zip(faces, bounding_boxes):
                
                left  = int(bb[0] + 10)
            
                up    = int(bb[1] - 10)
                
                gender = gendernet.predict([face])[0]
                
                age = agenet.predict([face])[0]
                
                text = str(gender) + ' ' +str(round(age,0)) 
                
                text_location = (left+5, up+2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                cv2.putText(new_frame, text, text_location, font, 1,(255,255,255),2)
                
                print(str(gender) + ' ' +str(round(age,4)) )
        
        tim2 = time.time()
        
        print('FPS = {}'.format( 1.0/(tim2-tim1) ))
        
#        cv2.imshow('1', frame)
        
        cv2.imshow('2', new_frame)
        
        key = cv2.waitKey(50)
        
        if 27==key:
            
            break
                
        ret, frame = cap.get_frame()
                
#        break
    
    cv2.destroyAllWindows()
    
    del cap
    
    