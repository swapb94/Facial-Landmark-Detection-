import numpy as np
import dlib
import cv2
import argparse
from collections import OrderedDict
import sys


#create dictionary to store the points which denote the specified region.
#for example points from 48(python uses zero indexing) to 54 represent the upper lip 
Dict = OrderedDict([("upper_lip",(48,55)),("right_eye",(36,42))])
#converts the rectangle returned by predictor to bounnding box that is a four element tuple:(x_coord,y_coord,width,height)
def four_tuple_transform(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y	
	return (x, y, w, h)
#to convert numpy array of ordered pairs
def tonumpy(shape,dtype = 'int'):
    coords = np.zeros((68,2),dtype=dtype)
    for i in range(0,68):
        coords[i] = (shape.part(i).x,shape.part(i).y)
    return coords
#takes care that even though if image is resized its aspect ratio is maintained
def resize(img,width):
	(h,w) = img.shape[:2]
	r = width/float(w)
	img = cv2.resize(img,(width,int(h*r)),interpolation=cv2.INTER_AREA)
	return img

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cam = cv2.VideoCapture(0) 	
#'0' is passed as the argument inorder to access the primary camera(inbuilt webcam)		
while(True):
	ret,image = cam.read();
	#image = cv2.imread('C:\\Users\\Prakash\\Desktop\\m\\dlib\\swap.jpg')
	image = resize(image,500)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	faces = detector(gray,1)
	alpha = 0.5
	#loop over all the faces found in the input frame
	for (i,face) in enumerate(faces):
		#prdict shape using dlib facial landmark pre-trained model
		shape = predictor(gray,face)
		shape = tonumpy(shape)
		top = image.copy()
		main = image.copy()
		#loop over all the shapes present in the dictionary
		for (i,key) in enumerate(Dict.keys()):
			(x,y) = Dict[key]
			#x,y denotes the starting point and ending point coordinate for particular shape
			points = shape[x:y]
			#'points' contains set of all points out of a total of 68 points which denote that particular area
			hull = cv2.convexHull(points)
			cv2.drawContours(top,[hull],-1,(0,0,255),-1)
		(x,y,w,h) = four_tuple_transform(face)
		cv2.rectangle(main,(x,y),(x+w,y+h),(0,255,0),2)
		f= faceDetect.detectMultiScale(gray,1.3,5)
		#ScaleFactor - 1.3,MinSize - 5X5
		for (x,y,w,h) in f:
			cv2.rectangle(main,(x,y),(x+w,y+h),(255,0,),2)
		#alpha represents the transparency factor while overlapping two images
		cv2.addWeighted(top,alpha,main,1-alpha,0,main)
		cv2.imshow("Image", main)
		#cv2.waitKey(0)
	if(cv2.waitKey(1)==ord('q')):
		break;
cam.release();
cv2.destroyAllWindows();


	    
