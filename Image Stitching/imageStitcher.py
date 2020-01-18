'''
File: imageStitcher.py
Author: John Hacker
Date: 12/8/2019

README:
    The functions used for SIFT were only created starting at OpenCV 3.0.0.0
and were deprecated after OpenCV 3.4.2.16. Therefore, you must have a version
between (inclusive) those numbers in order to execute this script.

    Images to be stitched must be in the directory identified as 
"imagesFolder" below. Images need to be named so that the alphabetical order 
and right-to-left order of position is the same. If the images are not 
ordered this way, this script will NOT work.

    The default values for "ratio" and "reprojThresh" may need to be 
adjusted for best results depending on the input. Typically, "ratio" should
be between 0.7 and 0.8 and "reprojThresh" between 1 and 10.

DESCRIPTION:
    This script will stitch several images together, so long as they are 
prepared appropriately (read above README). The prupose of this program is 
to show an understanding of SIFT and image stitching along with a solution 
of how to apply these methods to more than two images.

    All the steps of stitching will be shown as a pyplot upon completion of 
this script. Each stitched image will be saved to the directory identified
as "resultsFolder" below. 
'''

# Directories for input and output images
imagesFolder = "./Images3/"
resultsFolder = "./Results/"

# Import statements
import sys
import os
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

'''
Stitches two images together using SIFT
Inputs: two images, Lowe's ratio, and reprojection threshold
Output: stitched image
'''
def Stitch(img1, img2, ratio=0.75, reprojThresh=4.0, showMatches=False):
    # Find keypoints and descriptors in both image
    (kps1, descs1) = detectAndDescribe(img1)
    (kps2, descs2) = detectAndDescribe(img2)

    # Match features between the two images
    H = matchKeypoints(kps1, kps2, descs1, descs2, ratio, reprojThresh)

    # No keypoints matched --> images cannot be stitched
    if H is None:
        return None
    
    # Apply a perspective warp to stitch the images together
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    # Return the stitched image
    return result

'''
Performs SIFT on an image to find its keypoints and descriptors
Input: image
Ouptuts: keypoints and descriptors
'''
def detectAndDescribe(img):

    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(img, None)

    # convert the keypoints from KeyPoint objects to NumPy arrays
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, descs)

'''
Match the keypoints of two images using their descriptors
Inputs: two lists each of keypoints and descriptors, Lowe's ratio, and reprojection threshold
Output: homography matrix
'''
def matchKeypoints(kps1, kps2, descs1, descs2, ratio, reprojThresh):
    # Compute raw matches and initialize the list of actual matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(descs1, descs2, 2)
    matches = []

    # Loop over the raw matches
    for m in rawMatches:
        # Lowe's ratio test
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    
    # Check if there are more than 4 matches
    if len(matches) > 4:
        # Construct the two sets of points
        pts1 = np.float32([kps1[i] for (_, i) in matches])
        pts2 = np.float32([kps2[i] for (i, _) in matches])

        # Compute the homography between the two sets of points
        (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)

        # Return the homography matrix
        return H

    # No homograpy could be computed
    return None

# Get the image files to stitch
# * Must be alphabetical order from right to left *
imgFiles = os.listdir(imagesFolder)

# Loop through all the images
img1 = img2 = None
sbplt = 100 * (len(imgFiles) - 1) + 31
count = 1
for imgFile in imgFiles:

    # Path to image
    imgPath = imagesFolder + imgFile

    # Open the first image
    if img2 is None:
        print("First: " + imgFile)
        img2 = cv2.imread(imgPath)
        
    # Open the rest of the images and stitch with previous result/first pic
    else:
        # Copy the previous result or first picture
        print("Then: " + imgFile)
        img1 = img2

        # Plot the previous image
        img1_RGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        plt.subplot(sbplt), plt.imshow(img1_RGB)
        plt.title("image 1"), plt.xticks([]), plt.yticks([])
        sbplt += 1

        # Open the current image
        img2 = cv2.imread(imgPath)

        # Plot the current image
        img2_RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        plt.subplot(sbplt), plt.imshow(img2_RGB)
        plt.title("image 2"), plt.xticks([]), plt.yticks([])
        sbplt += 1

        # Stitch the previous and current images together
        img2 = Stitch(img1, img2)

        # Plot the resulting stitched image
        img2_RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        plt.subplot(sbplt), plt.imshow(img2_RGB)
        plt.title("stitched"), plt.xticks([]), plt.yticks([])
        sbplt += 1

        # Check if a result was possible
        if img2 is None:
            print("Images could not be stitched")
            break

        # Create the results directory if it does not exist
        if not os.path.exists(resultsFolder):
            os.makedirs(resultsFolder)

        # Save the resulting image
        fileName, fileExt = os.path.splitext(imgPath)
        resultFile = resultsFolder + "Stitched" + str(count) + fileExt 
        cv2.imwrite(resultFile, img2)
        count += 1

# Display the process of stitching all the images together
plt.show()
