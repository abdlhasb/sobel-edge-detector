#this is only for getting the current working directory
import os
# step 1 import libraries
from PIL import Image
from math import sqrt
import numpy as np

directoryPath = os.getcwd()
# Open color image
img = Image.open(f"{directoryPath}\\Q3_Input\Haseeb_23i7801.JPG")
# get the image size
width, height = img.size

#create a matrix for grayscale, Gx, Gy and edgeDetector
grayscale_image = []
Gx = []
Gy = []
edgeDetector = []

# this is a zero row for add zeros to the edges so our 3 x 3 window looks correct
zeroRow = [0]*(width+2)
# for adding 0 row and 0 column
grayscale_image.append(zeroRow)
Gx.append(zeroRow)
Gy.append(zeroRow)
edgeDetector.append(zeroRow)

# loop through the height
for y in range(height):
    myRow = [0]
    for x in range(width):
        r, g, b = img.getpixel((x, y))
        # Calculate the grayscale value using the formula (this formula has a better scalling than the average formula)
        grayscale_value = int((0.2989 * r) + (0.5870 * g) + (0.1140 * b))
        myRow.append(grayscale_value)
    myRow.append(0)
    grayscale_image.append(myRow)

#append the last zero row so our matrix will work for the edge detection
grayscale_image.append(zeroRow)

# remove the zeros on the boundaries of image and display a grayscale image
trimmed_matrix = [row[1:-1] for row in grayscale_image[1:-1]]
g_Image = Image.fromarray(np.uint8(trimmed_matrix) , 'L')
g_Image.save(f"{directoryPath}\\Q3_Output\Haseeb_23i7801_Grayscale_Image.png")
# g_image_show = Image.open(os.getcwd()+"\\"+"gray_scale_image.png")
# g_image_show.show()

#kernals
xKernel = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]
yKernel = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]

# compute Gx pixel matrix value with xKernel
edgeSum = 0
matrix = []
for i in range(1,height+1,1): #remove the zeros row and column
    myRow = [0]
    for j in range(1,width+1,1):
        for _row in range(-1, 2):
            pixel_value = []
            for _col in range(-1, 2):
                pixel_value.append(grayscale_image[i + _row][j + _col])
            matrix.append(pixel_value)
        
        # matrix multiplication window x kernels
        for _row in range(0,3):
            for _col in range(0,3):
                edgeSum += (matrix[_row][_col]*xKernel[_row][_col])
        myRow.append(edgeSum)
        edgeSum = 0
        matrix = []
    myRow.append(0)
    Gx.append(myRow)
    
Gx.append(zeroRow)

# compute Gy pixel matrix value with yKernel
for i in range(1,height+1,1):
    myRow = [0]
    for j in range(1,width+1,1):
        for _row in range(-1, 2):
            pixel_value = []
            for _col in range(-1, 2):
                pixel_value.append(grayscale_image[i + _row][j + _col])
            matrix.append(pixel_value)
                
        for _row in range(0,3):
            for _col in range(0,3):
                edgeSum += (matrix[_row][_col]*yKernel[_row][_col])
        myRow.append(edgeSum)
        edgeSum = 0
        matrix = []
    myRow.append(0)
    Gy.append(myRow)
    
Gy.append(zeroRow)

# compute edgeDetector value with respect to Gx and Gy
for i in range(1,height+1,1):
    myRow = [0]
    for j in range(1,width+1,1):
        detectorValue = (Gx[i][j]**2) + (Gy[i][j]**2)
        sqrtDetectorValue = int(sqrt(detectorValue))
        # threshhold
        if sqrtDetectorValue > 255:
            sqrtDetectorValue = 255
        myRow.append((sqrtDetectorValue))
    myRow.append(0)
    edgeDetector.append(myRow)
    myRow = []
edgeDetector.append(zeroRow)

# trim the boundaries zeros that I add to compute the padding pixels
trimmed_matrix = [row[1:-1] for row in edgeDetector[1:-1]]
final_image = Image.fromarray(np.uint8(trimmed_matrix), 'L')
final_image.save(f"{directoryPath}\\Q3_Output\Haseeb_23i7801_EdgeDetection_Image.png")
edge_image_show = Image.open(f"{directoryPath}\\Q3_Output\Haseeb_23i7801_EdgeDetection_Image.png")
edge_image_show.show()