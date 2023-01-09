import cv2 as cv
import numpy as np
import pandas as pd
from decimal import Decimal
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

path = 'Plant_Images'
files=os.listdir(path)
images=[]
labels=[]
aspect_ratios=[]
variable=1
var_=1

for filename in files:
    if filename.endswith(('.JPG', '.jpg')):
        image_name = os.path.join(path, filename)
        image = cv.imread(image_name)
        images.append(image)
        labels.append(filename)
        # Extract label from file name
        # labels.append(filename)

        # Converting image to create binary image
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Mask()
        lower_green = np.array([52, 0, 55])
        upper_green = np.array([104, 255, 255])
        mask = cv.inRange(image_hsv, lower_green, upper_green)

        # Output after masking
        output_hsv = image_hsv.copy()
        output_hsv[np.where(mask == 0)] = 0

        # Conversion to grayscale
        gray = cv.cvtColor(output_hsv, cv.COLOR_BGR2GRAY)

        # Histogram Equalization
        equal_histogram = cv.equalizeHist(gray)

        # Applying Gaussian Blur
        blur = cv.GaussianBlur(gray, (3, 3), 0)

        # Thresholding the image to create binary Image
        ret, binary = cv.threshold(blur, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        print("No. of contours detected -> %d " % len(contours))

        contours = sorted(contours, key=cv.contourArea, reverse=True)[:1]
        screen_contours = None

        for c in contours:
            # Finding the bounding box of the leaf
            x, y, w, h = cv.boundingRect(binary)

            # Drawing the bounding reactangle on the image
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the aspect ratio of the leaf
            a_ratio = float(w) / h

            print("Aspect ratio", a_ratio)
            aspect_ratios.append(a_ratio)

        file_name = "D:\AI_CW\demo\leaves\leaf_" + str(variable) + ".jpg"
        print(file_name)
        variable += 1
        cv.imwrite(file_name, image)


x = images
y = aspect_ratios

print(labels)
d1 = {'mango': 1, 'basil': 2, 'chinar': 3}
digits = []
for label in labels:
    digit = int(label[:4])

    digits.append(digit)

print(digits)
print(len(digits))

#Split data into Training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

#en(X_train),len(X_test)

#clf = RandomForestClassifier(random_state=1)
#print(clf)

#clf.fit(X_train,y_train)

# Make predictions on the test data
#predictions = model.predict(X_test)

# Evaluate the model
#accuracy = model.score(X_test, y_test)
#print("Accuracy:", accuracy)