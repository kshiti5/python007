import cv2
from random import randrange

# Load Some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    './haarcascade_frontalface_default.xml')

# step1 -  Choose an image to detect face in
img = cv2.imread("tejas.jpeg")


# step2 - Convert to Grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# step3 - Detect face
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
# print(face_coordinates)

# Draw rectangle around face
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cv2.rectangle(img, (307, 115), (336+336, 115+336), (0, 255, 0), 2)


''''
    If we put these code in loop it will covered the multiple face.
# '''

# (x,y,w,h) = face_coordinates[0]
# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Just like

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
                  randrange(256), randrange(256)), 2)


cv2.imshow("Tejas Agrawal", img)
cv2.waitKey()

print("Code Compete")
