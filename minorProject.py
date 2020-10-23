import numpy as np
import cv2
import tensorflow as tf

sta = False
img = np.ones([300,300])*255
windowName='DigitRecognition'
cv2.namedWindow(windowName)
m_new=tf.keras.models.load_model(r'C:\Users\hp\Downloads\model_digitRecog.h5')


def demo(event,x,y,flags,param):
    global sta
    if event==cv2.EVENT_LBUTTONDOWN:
        sta= True
    elif event==cv2.EVENT_MOUSEMOVE:
        if sta == True:
            cv2.circle(img,(x,y),8,(0,0,0),-5)
    elif event==cv2.EVENT_LBUTTONUP:
        sta= False

   
                  
cv2.setMouseCallback(windowName,demo)
while(True):
       cv2.imshow(windowName,img)

       if cv2.waitKey(1)==ord('c'):
          img[:,:]=255

       elif cv2.waitKey(1)==ord('p'):
           out= img[:,:]
           image_test_resize=cv2.resize(out,(28,28)).reshape(1,28,28)
           print(m_new.predict_classes(image_test_resize))
       elif cv2.waitKey(1)== ord('q'):
           break
           
cv2.destroyAllWindows()  
