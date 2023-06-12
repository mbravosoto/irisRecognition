# import the necessary packages

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import feature
from iris_recognition import *


class LocalBinaryPatterns:

    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist, lbp

if __name__ == "__main__":

    mypath = 'C:/Users/Juan Pablo/Im_Procesamiento/archive/MMU-Iris-Database TrainTest/'
    images_list = []

    # Read paths images from the dataset folder
    for root, dirs, files in os.walk(mypath):
        for file in files:
            if file.endswith("1.bmp"):
                images_list.append(os.path.join(root, file))

    # initialize the local binary patterns descriptor along with
    # the data and label lists

    desc = LocalBinaryPatterns(3, 3) # default: 24,8
    data = []
    labels_svc = []
    persona = 1

    for imagePath in images_list:

        # load the image, convert it to grayscale, and describe it

        image = cv2.imread(imagePath)

        ### IRIS RECOGNITION ###
        iris = iris_recognition(image)
        polar_image, _, _ = iris.save()

        width, height = polar_image.shape
        croped_polar = polar_image[0:width // 2, :]
        croped_polar_eq = cv2.equalizeHist(croped_polar)

        #output the lbp hist and image
        hist, lbp = desc.describe(croped_polar_eq)


        # extract the label from the image path, then update the
        # label and data lists
        print(persona)
        labels_svc.append(str(persona))  # os.path.split(os.path.dirname(imagePath))[-1])
        persona += 1
        data.append(hist)

        # show images
        cv2.imshow("croped polar eq", croped_polar_eq)
        cv2.imshow("cropped polar", croped_polar)
        cv2.imshow("blp of iris image", lbp)
        iris.show()
        cv2.waitKey(0)

    plt.plot(data)
    plt.show()

