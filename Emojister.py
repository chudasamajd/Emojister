import cv2
import numpy as np
from PIL import Image




def nothing(x):
    pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/Trained_model.h5')

def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][5] == 1:
              return 'F'
       elif result[0][6] == 1:
              return 'G'
       elif result[0][11] == 1:
              return 'L'
       elif result[0][16] == 1:
              return 'Q'
       elif result[0][17] == 1:
              return 'R'
       elif result[0][18] == 1:
              return 'S'
       
       elif result[0][21] == 1:
              return 'V'
       elif result[0][24] == 1:
              return 'Y'
       
       

       

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")


    img = cv2.rectangle(frame, (425,100),(625,300), (207,255,51), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    #cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))

    # Make PIL image from frame, paste in speedo, revert to OpenCV frame
    if img_text == 'F':
            speedo = Image.open('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/emoji/F.png').convert('RGBA')
            R,G,B,A = speedo.split()
            speedo = Image.merge('RGBA',(B,G,R,A))

            pilim = Image.fromarray(frame)
            pilim.paste(speedo,box=(200,300),mask=speedo)
            frame = np.array(pilim)
    elif img_text == 'G':
            speedo = Image.open('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/emoji/G.png').convert('RGBA')
            R,G,B,A = speedo.split()
            speedo = Image.merge('RGBA',(B,G,R,A))

            pilim = Image.fromarray(frame)
            pilim.paste(speedo,box=(200,300),mask=speedo)
            frame = np.array(pilim)
    elif img_text == 'L':
            speedo = Image.open('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/emoji/L.png').convert('RGBA')
            R,G,B,A = speedo.split()
            speedo = Image.merge('RGBA',(B,G,R,A))

            pilim = Image.fromarray(frame)
            pilim.paste(speedo,box=(200,300),mask=speedo)
            frame = np.array(pilim)
    elif img_text == 'Q':
            speedo = Image.open('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/emoji/Q.png').convert('RGBA')
            R,G,B,A = speedo.split()
            speedo = Image.merge('RGBA',(B,G,R,A))

            pilim = Image.fromarray(frame)
            pilim.paste(speedo,box=(200,300),mask=speedo)
            frame = np.array(pilim)
    elif img_text == 'R':
            speedo = Image.open('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/emoji/R.png').convert('RGBA')
            R,G,B,A = speedo.split()
            speedo = Image.merge('RGBA',(B,G,R,A))

            pilim = Image.fromarray(frame)
            pilim.paste(speedo,box=(200,300),mask=speedo)
            frame = np.array(pilim)
    elif img_text == 'S':
            speedo = Image.open('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/emoji/S.png').convert('RGBA')
            R,G,B,A = speedo.split()
            speedo = Image.merge('RGBA',(B,G,R,A))

            pilim = Image.fromarray(frame)
            pilim.paste(speedo,box=(200,300),mask=speedo)
            frame = np.array(pilim)
    
    elif img_text == 'V':
            speedo = Image.open('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/emoji/V.png').convert('RGBA')
            R,G,B,A = speedo.split()
            speedo = Image.merge('RGBA',(B,G,R,A))

            pilim = Image.fromarray(frame)
            pilim.paste(speedo,box=(200,300),mask=speedo)
            frame = np.array(pilim)
    elif img_text == 'Y':
            speedo = Image.open('D:/Python Projects/Test/Simple-Sign-Language-Detector-master/Simple-Sign-Language-Detector-master/emoji/Y.png').convert('RGBA')
            R,G,B,A = speedo.split()
            speedo = Image.merge('RGBA',(B,G,R,A))

            pilim = Image.fromarray(frame)
            pilim.paste(speedo,box=(200,300),mask=speedo)
            frame = np.array(pilim)

    cv2.imshow("test", frame)
    cv2.imshow("mask", mask)
    
    #if cv2.waitKey(1) == ord('c'):
        
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    
    img_text = predictor()
        

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()