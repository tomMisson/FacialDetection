import cognitive_face as CF
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib

def url_to_image(url):
  resp = urllib.request.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  # return the image
  return image

KEY = '15b61d3dd99f470893c816150ec4a5f9'  # Replace with a valid Subscription Key here.
CF.Key.set(KEY)

BASE_URL = 'https://uksouth.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

img_url = 'http://facedetect.ddns.net/Images/img5.jpg'
result = CF.face.detect(img_url)
img = url_to_image(img_url)

height = result[0]['faceRectangle']['height']
left = result[0]['faceRectangle']['left']
top = result[0]['faceRectangle']['top']
width = result[0]['faceRectangle']['width']

plt.imshow(cv2.cvtColor(img[top:top+height, left:left+width], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
 

"""
cap = result

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('Original',frame)
    edges = cv2.Canny(frame,100,200)
    cv2.imshow('Edges',edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
"""