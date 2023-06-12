import os
import cv2
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.cluster import KMeans


class iris_recognition:

    def __init__(self,image):

        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.height, self.width, _ = self.image.shape
        self.mask = np.zeros((self.height, self.width), np.uint8)

        imgi = self.image.copy()
        imgp = self.image.copy()
        self.output = self.image.copy()

        # ajustes preliminares para la implementacion de los algoritmos de clasisficación

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_rgb = np.array(self.image, dtype=np.float64) / 255
        rows, cols, ch = self.image_rgb.shape

        # K MEANS AND BLUR

        assert ch == 3
        image_array = np.reshape(self.image_rgb, (rows * cols, ch))
        image_array_sample = shuffle(image_array)[:10000]

        ncolors = 10
        model = KMeans(n_clusters=ncolors).fit(image_array_sample)
        labels = model.predict(image_array)
        centers = model.cluster_centers_
        km = iris_recognition.recreate_image(centers, labels, rows, cols)

        # convert array to image
        km_ = np.floor(km * 255)
        juanpis = km_.astype(np.uint8)
        manu = Image.fromarray(juanpis)
        manu.save("eye1.jpg")
        gray_ = cv2.imread('eye1.jpg')
        gray_ = cv2.cvtColor(gray_, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray_, 7)

        if os.path.exists("eye1.jpg"):
            os.remove("eye1.jpg")

        # PUPIL DETECTION

        img = gray_blur.copy()
        im_gray = gray_blur.copy()
        treshold = np.floor(np.max(im_gray) * 0.15)

        ret, im_bw = cv2.threshold(im_gray, treshold, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((9, 9), np.uint8)
        pupil = im_bw.copy()

        pupil = cv2.morphologyEx(pupil, cv2.MORPH_OPEN, kernel)
        pupil = cv2.morphologyEx(pupil, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(pupil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        cy = 0
        cx = 0

        if contours:
            M = cv2.moments(contours[0])
            area = M['m00']
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            radius = int(np.ceil(np.sqrt(area / (2 * np.pi))))
            (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
            center = (int(cx), int(cy))
            radius = int(radius)
            cv2.circle(self.output, center, radius, (0, 0, 255), 2)
            cv2.circle(imgp, center, radius, (0, 0, 0), -1)

        # IRIS DETECTION WITH HOUGHS CIRCLES

        height1, width1 = gray_blur.shape

        # detect circles in the image with Hough
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1.5, width1 / 4, param1=100, param2=30,
                                   minRadius=40, maxRadius=80)
        # ensure at least some circles were found
        if circles is not None:
            if cx > 0 and cy > 0:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(self.output, center, r, (0, 255, 0), 4)
                    cv2.circle(imgi, center, r, (0, 0, 0), -1)
                    break

            else:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(self.output, (x, y), r, (0, 255, 0), 4)
                    cv2.circle(imgi, (x, y), r, (0, 0, 0), -1)
                    break
        else:
            print("not iris founded")

        # IRIS EXTRACTION
        ret, im_bw = cv2.threshold(imgi, treshold, 255, cv2.THRESH_BINARY_INV)
        self.mask = cv2.bitwise_and(im_bw, imgp)


        # POLAR COORDINATES
        img = self.mask.astype(np.float32)
        radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))

        if cx > 0 and cy > 0:
            polar_image = cv2.linearPolar(img, (cx, cy), radius, cv2.WARP_FILL_OUTLIERS)
        else:
            polar_image = cv2.linearPolar(img, (x, y), radius, cv2.WARP_FILL_OUTLIERS)

        polar_image = polar_image.astype(np.uint8)
        polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_CLOCKWISE)
        polar_image = cv2.cvtColor(polar_image, cv2.COLOR_BGR2GRAY)
        self.polar_image = iris_recognition.crop_image(polar_image, tol=80)


    # metodo para recrear la imagen ingresada aplicando la clasificación del metodo
    @staticmethod
    def recreate_image(centers, labels, rows, cols):
        d = centers.shape[1]
        image_clusters = np.zeros((rows, cols, d))
        label_idx = 0
        for i in range(rows):
            for j in range(cols):
                image_clusters[i][j] = centers[labels[label_idx]]
                label_idx += 1
        return image_clusters

    # metodo para recortar los bordes negro de una imagen
    @staticmethod
    def crop_image(image, tol=0):
        # method  form: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934
        # img is 2D image data
        # tol  is tolerance
        mask_ = image > tol
        return image[np.ix_(mask_.any(1), mask_.any(0))]

    def show(self):
        cv2.imshow("output", self.output)
        cv2.imshow('iris separation', self.mask)
        cv2.imshow("Polar Image", self.polar_image)

    def save(self):
        return self.polar_image, self.output, self.mask

