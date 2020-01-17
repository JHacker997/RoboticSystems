'''
Author: John Hacker
Course: EEL 4660
Semester: Fall 2019
Python: 3.6.8
OpenCV: 4.1.1
'''

import cv2
import os
import numpy as np

# Input:
#     method:       method of object detection
#     files:        list of all file names
#     qfile:        name of query file
#     imgs:         list of all cv2 images
#     qImg:         cv2 image to query
# Output:
#     correct:      number of correct top-four matches
#     topScores[1]: list offile names for the top-four matches
def ObjectDetection(method, files, qfile, imgs, qImg):
    # Precompute the histogram of the query image
    if method == 'compHist':
        queryHist = cv2.calcHist([qImg], [0], None, [256], [0,256])
    
    # Perform object detection of each image using the query image
    topScores = [[0,0,0,0],['','','','']]
    for i, img in enumerate(imgs):
        if method == 'compHist':
            imgHist = cv2.calcHist([img], [0], None, [256], [0,256])
            newScore = cv2.compareHist(queryHist, imgHist, 2)
        elif method == 'matchTemp':
            result = cv2.matchTemplate(img, qImg, eval('cv2.TM_CCOEFF'))
            newScore = cv2.minMaxLoc(result)[1]
        
        # If the match score is higher than one of the top-four, replace it
        if newScore > min(topScores[0]):
            index = topScores[0].index(min(topScores[0]))
            topScores[0][index] = newScore
            topScores[1][index] = os.path.splitext(files[i])[0]

    # Sort the top-four matches by match score
    indices = [0,1,2,3]
    indices.sort(key = topScores[0].__getitem__, reverse=True)
    for i, sublist in enumerate(topScores):
        topScores[i] = [sublist[j] for j in indices]
    
    # Count how many top-four matches are actually the same class as the query
    correct = 0
    for i, fileName in enumerate(topScores[1]):
        queryNum = int(os.path.splitext(qfile)[0][-5:])
        fileNum = int(fileName[-5:])
        if queryNum - 4 <= fileNum <= queryNum + 4:
            correct = correct + 1

    return correct, topScores[1]

# Define the image classes and methods to compare
classes = ["duck", "chair", "girl", "painting", "photos"]
methods = ['compHist', 'matchTemp']

# Read all of the images to test on
DIR = 'images'
files = os.listdir(DIR)
imgs = []
for fileName in files:
    if not fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    imgs.append(cv2.imread(DIR + "/" + fileName, cv2.IMREAD_GRAYSCALE))

# Query each of the images using both methods
histScores = []
tempScores = []
for i, img in enumerate(imgs):
    histNumCorrect, histTopRanking = ObjectDetection(methods[0], files, files[i], imgs, img)
    histScores.append(histNumCorrect)
    tempNumCorrect, tempTopRanking = ObjectDetection(methods[1], files, files[i], imgs, img)
    tempScores.append(tempNumCorrect)

    # Report how the two queries went
    print("class: " + classes[int(i/4)])
    print("query: " + files[i])
    print("score_color_hist: " + str(histNumCorrect))
    print("score_template_match: " + str(tempNumCorrect))
    print("top4_color_hist: " + str(histTopRanking))
    print("top4_template_match: " + str(tempTopRanking))
    print()

# Report statistics on all the queries
meanDict = [["color_histogram_mean", np.mean(histScores)], ["Template_match_mean", np.mean(tempScores)]]
stdDevDict = [["color_histogram_stdDev", np.std(histScores)], ["Template_match_stdDev", np.std(tempScores)]]
print(dict(meanDict))
print(dict(stdDevDict))
