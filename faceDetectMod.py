import cv2
import sys


# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]
faceImagePath = sys.argv[3]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceImage = cv2.imread(faceImagePath, -1)
S = (0.5, 0.5, 0.5, 0.5)			# Define blending coefficients S and D
D = (0.5, 0.5, 0.5, 0.5)

def overlay(image, faceImage, posx, posy, S, D):
    for x in range(faceImage.shape[0]):
        if x + posx < image.shape[0]:
            for y in range(faceImage.shape[1]):
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
 

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #head = image[y: y+h,x: x+w] 
    overlay(image, faceImage, x, y, S, D)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
