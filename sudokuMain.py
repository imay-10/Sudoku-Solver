print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
import sudokuSolver

########################################################################
pathImage = "Resources/3.png"
heightImg = 324
widthImg = 324
model = intializePredectionModel()  # LOAD THE CNN MODEL
########################################################################



##### 1. PREPARE THE IMAGE

img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8) 
imgThreshold = preProcess(img)



##### 2. FIND ALL COUNTOURS

imgContours = img.copy() 
imgBigContour = img.copy() 
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)



##### 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU

biggest, maxArea = biggestContour(contours)
print(biggest)
if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 15)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)



    #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)
    print(len(boxes))
    # cv2.imshow("Sample",boxes[55])
    numbers = getPrediction(boxes, model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)



    ##### 5. FIND SOLUTION OF THE BOARD

    board = np.array_split(numbers,9)
    print(board)
    try:
        sudokuSolver.solve(board)
    except:
        pass
    print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList*posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)



    ##### 6. OVERLAY SOLUTION

    pts2 = np.float32(biggest)
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    # imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)


else:
    print("No Sudoku Found")

cv2.waitKey(0)
