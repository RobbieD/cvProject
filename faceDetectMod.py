import cv2
import sys


# Get user supplied values
imagePath = sys.argv[1]

faceImagePath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceImage = cv2.imread(faceImagePath, -1)

cap = cv2.VideoCapture(0)

print(faceImage.shape)

S = (0.0, 0.0, 0.0, 0.0)			# Define blending coefficients S and D
D = (1, 1, 1, 1)



def overlay(image, faceImage, posx, posy, S, D, w, h):
    #print (" posx: "+str(posx)+" posy: "+str(posy)   )
    faceImage = cv2.resize(faceImage, (w, 2*h)	)
    for x in range(faceImage.shape[1]):
        if x + posx < image.shape[1]:
            for y in range(faceImage.shape[0]):
                if y + posy < image.shape[0]:
                    source = cv2.cv.Get2D(cv2.cv.fromarray(image), y+posy, x+posx)
                    over = cv2.cv.Get2D(cv2.cv.fromarray(faceImage), y, x)
                    merger = [0, 0, 0, 0]

                    for i in range(3):
                        if over[i] == 0:
                            merger[i] = source[i]
                        else:
                            merger[i] = (S[i]*source[i]+D[i]*over[i])
                    merged = tuple(merger)
                    
                    cv2.cv.Set2D(cv2.cv.fromarray(image), y+posy, x+posx, merged)
          #      else:
           #         print("y too big?")
        #else:
         #   print("x too big?")

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    #print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            #detect eyes for scaling purposes
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        overlay(frame, faceImage, int(x - (0.2 * w)), int(y - (0.2 * h)), S, D, w, h)

   	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#cv2.imshow("Faces found", image)
#cv2.waitKey(0)

		




