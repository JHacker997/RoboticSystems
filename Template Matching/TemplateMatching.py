# Import appropriate libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

def MatchTemplate(img, waldo, num):
    # Dimensions of the query image
    height, width = waldo.shape

    # Convert the image to gray scale (needed for matchTemplate)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the image to RGB to plot and show/save
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Array of different template matching methods
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # Loop through each template matching method
    for method in methods:

        # Create a copy of the RGB image to draw a rectangle on later
        imgCopy = img.copy()
        imgRGBCopy = imgRGB.copy()

        # Instantiate the appropriate evaluation method
        evaluation = eval(method)

        # Find where the template is in the image
        res = cv2.matchTemplate(imgGray, waldo, evaluation)
        min_loc, max_loc = cv2.minMaxLoc(res)[2:4]

        # Determine the top left corner of the bounding box to the template
        if evaluation in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        # Determine the bottom right corener of the bounding box of the template
        bottom_right = (top_left[0] + width, top_left[1] + height)

        # Draw the bounding box on the RGB and RBG image where the template was found
        cv2.rectangle(imgCopy, bottom_right, top_left, (255, 0, 255), 2)
        cv2.rectangle(imgRGBCopy, bottom_right, top_left, (255, 0, 255), 2)

        # Plot the reult and the detected match
        methodName = method.replace("cv2.", "")
        plt.subplot(121), plt.imshow(res, cmap = "gray")
        plt.title(methodName), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(imgRGBCopy)
        plt.title("Detected Result"), plt.xticks([]), plt.yticks([])

        # Save the plot
        fileName = r"solutions\\" + methodName + "Plot" + "_" + str(num) + ".png"
        plt.savefig(fileName, bbox_inches='tight', pad_inches = 0)

        # Display the plot
        plt.show()

        # Show and save the RBG image with the detected box drawn on it
        fileName = r"solutions\\" + methodName + "_" + str(num) + ".png"
        cv2.imshow(fileName, imgCopy)
        cv2.imwrite(fileName, imgCopy)
        cv2.waitKey(0)

# Open the first puzzle and query
img1 = cv2.imread("images\puzzle_1.jpg", 1)
waldo1 = cv2.imread("images\query_1.jpg", 0)

# Solve the first puzzle
MatchTemplate(img1, waldo1, 1)

# Open the second puzzle and query
img2 = cv2.imread("images\puzzle_2.png", 1)
waldo2 = cv2.imread("images\query_2.png", 0)

# Solve the second puzzle
MatchTemplate(img2, waldo2, 2)