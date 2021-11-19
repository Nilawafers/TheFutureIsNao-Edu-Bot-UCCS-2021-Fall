'''

@author: Nicholas Buckley
'''

import cv2

from cntk.device import cpu, try_set_default_device
from pip._vendor.html5lib._utils import _x
try_set_default_device(cpu())

import keras

import random

import numpy as np
#from matplotlib import pyplot as plt 

from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D

import cntk








#cut page into 5 levels based on the height of the page then sort left to right
def sort_Reading_Order(contour, cols, height):
    tolerance_factor = height/5.2 #5 levels on the page since 5 problems
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


# image rotation for possible re-prediction or data augmentation
def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img




def crop_Nao_Page(image):
    original_image = image
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #edges= cv2.Canny(gray, 50,200)
    
    
    
    # detect Nao Landmarks on the corners of the NaoMathPage
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=150, param2=30, minRadius=20, maxRadius=40)
    
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # loop over the (x, y, r) coordinates of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        
        #cv2.imshow(str(k), np.hstack([original_image]))
        #cv2.waitKey(0)



    
    count = 0
    xLeft = 960
    xRight = 0
    yBot = 1280
    yTop = 0
    
    # Crop Nao Paper out of the image based on the mark locations 
    for (x, y, r) in circles:
        
        
        
        if xLeft >  x:
            xLeft = x + r #+ w
        
        if xRight < x:
            xRight = x - r - 4#- w//2
            
        if yBot >  y:
            yBot = y + r - 8 #+ h
        
        if yTop < y:
            yTop = y - r + 6
            
        
        count +=1
        
        if count == 4:
            
            cropped_Paper = gray[yBot:yTop, xLeft:xRight]
            #image_name= "output_shape_number_" + str(i+1) + ".jpg"
            #cv2.imwrite(image_name, cropped_Paper)
            #readimage= cv2.imread(image_name)
            cv2.imshow('crop', cropped_Paper)
            cv2.waitKey(0)
            return cropped_Paper




def trainNewModel(name):
    
    # could be changed to any character data set 
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 
    
    
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  #28x28
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    #Normalising/Feature scaling
    X_train /= 255
    X_test /= 255
    
    
    nb_classes = 10 # number of unique digits
    Y_train = np_utils.to_categorical(y_train, nb_classes) 
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    
    # Building a fully connected (FC) network
    
    
    #The Glorot uniform initializer, also called He Normal initializer
    initializer = keras.initializers.he_normal()
    
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32,(3,3), padding="same"))
    model.add(Activation('relu'))
    
    
    
    model.add(Conv2D(64,(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(64,(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(Flatten())
    
    model.add(Dense(256, kernel_initializer=initializer)) #(784,) represents a 784 length vector
    model.add(Activation('relu'))   #sigmoid --> relu
    
    model.add(Dense(128, kernel_initializer=initializer))
    model.add(Activation('relu'))
    
    
    
    model.add(Dense(64, kernel_initializer=initializer))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    
    
    model.add(Dense(10, kernel_initializer=initializer))
    model.add(Activation('softmax'))
    model.summary()
    
    # Adam optimizer for learning
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=128, epochs=30, verbose=1)
    
    
    #model = keras.models.load_model("modelTest1")
    
    
    score = model.evaluate(X_test, Y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model.save(name)
    model.summary()
    print("Model : ", name, " is saved")
    
    
    
    
    
    

TOTALACC = 0.0
TOTALNUM = 21.0

testPage1True = [1,5,2,7,2,3,3,6,3,2,2,4,4,1,3,4,5,7,9,1,6]
testPage2True = [1,0,7,7,2,3,1,4,3,6,6,1,2,4,3,8,1,1,1,0,1]

testPage3True = [1,1,4,5,2,2,1,3,3,9,8,1,7,4,4,4,8,5,7,2,9]

testPage4True = [1,9,7,1,4,2,2,5,8,3,7,1,9,4,2,9,1,3,5,3,2,7]
testPage5True = [1,2,5,7,2,7,7,1,4,3,9,6,1,5,4,9,9,1,8,5,8,4,1,2]


imagname = "NaoPage3TestImage"
realPredictions = testPage3True



for p in range(21):
    
    
    image= cv2.imread(imagname + str(p) + '.png')
    cropped_Paper = crop_Nao_Page(image)
    
    
    ''' find circle based on number of segments and area
    contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    
    
    
    
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 10) & (area > 1000) ):
            contour_list.append(contour)
    
    
    cv2.drawContours(image, contour_list,  -1, (255,0,0), 2)
    cv2.imshow('Objects Detected',image)
    cv2.waitKey(0)
    '''
    
    #Extract numbers from the cropped page
    
    # order can be changed to effect performance
    edges= cv2.Canny(cropped_Paper, 105,200)
    
    # min threshold is dependent on lighting and does effect prediction performance heavily 
    (thresh, cropped_Paper) = cv2.threshold(cropped_Paper, 100, 255, cv2.THRESH_BINARY_INV) #Black and white
    contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    
    
    
    # Order in reading order
    contours.sort(key=lambda x:sort_Reading_Order(x, cropped_Paper.shape[1],cropped_Paper.shape[0] ))
    
    
    
    
    
    
    # Labeling the order on the cropped paper
    '''
    for i in xrange(len(contours)):
        img = cv2.putText(cropped_Paper, str(i), cv2.boundingRect(contours[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [125])
    cv2.imshow('magic',img)
    
    cv2.waitKey(0)
    '''


    saved = []
    numArray = np.zeros((800,28,28))
    
    
    
    
    x_ = 0
    k = 0
    for (i,c) in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        cropped_contour = cropped_Paper[y-1:y+h+2, x-1:x+w+2]
        
        xdiff = abs(x - x_)
        
        # 13 to 10 based on distance to filer our non numbers
        if h > 12 and xdiff > 3 : #small filter and overlapped contours filter
            old_image_height, old_image_width = cropped_contour.shape
            
            
            # 20 to 25 changes performance again bases on distance from the bot
            while old_image_height > 23 or old_image_width > 23:
                scale_percent = 85 # percent of original size
                old_image_width = int(cropped_contour.shape[1] * scale_percent / 100)
                old_image_height = int(cropped_contour.shape[0] * scale_percent / 100)
                dim = (old_image_width, old_image_height)
                # resize image
                #print(old_image_width, "    ",old_image_height )
                cropped_contour = cv2.resize(cropped_contour, dim, interpolation = cv2.INTER_AREA)
                
                
                
            # create new image of desired size and color for padding
            new_image_width = 28
            new_image_height = 28
            color = (0) #0 Black
            result = np.full((new_image_height,new_image_width), color, dtype=np.uint8)
            
            # compute center offset
            x_center = (new_image_width - old_image_width) // 2
            y_center = (new_image_height - old_image_height) // 2
            
            # copy img image into center of result image
            result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = cropped_contour
            image_name= "output_" + str(k+1) + ".jpg"
            saved.append(result)
            
            
            
            
            numArray[k] = result
            
            k += 1
            #cv2.imshow("Resized image", result)
            #cv2.waitKey(0)
            
            x_ = x
            
            
            
    
    
    numArray = numArray.astype('float32')
    numArray = numArray.reshape(numArray.shape[0], 28, 28, 1)
    
    
    model = keras.models.load_model("modelTest1")
    
    #model.summary()
    
    
    predictions = model.predict_on_batch(numArray)
    numOfNums = len(realPredictions)
    acc = 0.0
    tot = 0.0
    
    rawPredictions = []
    outputProbPred = [ ]
    outputAnsPred =  [[], [], [], [], []]
    AnsFromProb = []
    probNumsSecondCheck = False
    
    for (i, image) in enumerate(saved):
        #print("Top Prediction: ", np.argmax(predictions[i]))
        
        rawPredictions.append(np.argmax(predictions[i]))
        
        
        
        
        tot +=1
        if np.argmax(predictions[i]) == realPredictions[i%numOfNums]:
            acc +=1
        else:
            print("Top Prediction: ", np.argmax(predictions[i]))
            #cv2.imshow("Missed",image)
            #cv2.waitKey()
        
        #top3 = np.argpartition(predictions[i], -3)[-3:]
        #print("Top 3 Predictions: ", top3[np.argsort(predictions[i][top3])])
        
        #print("Top 3 values: ",np.array(predictions[i][top3], dtype='double'))
        if i%(numOfNums-1) == 0 and i!=0:
            print(float(acc/numOfNums))
            TOTALACC += float(acc/numOfNums)
        
        
    
    
    cv2.destroyAllWindows()
    
    probIndex = 1
    for i in range(5):
        
        outputProbPred.append(rawPredictions[probIndex:probIndex+2])
        
        AnsFromProb.append(outputProbPred[i][0] + outputProbPred[i][1])
        
        
        if probIndex + 3 < len(rawPredictions):
            if rawPredictions[probIndex + 3] == i + 2:
                probNumsSecondCheck = False
            else:
                probNumsSecondCheck = True
        
        
        if AnsFromProb[i] >= 10 and probNumsSecondCheck :
            outputAnsPred[i].append(rawPredictions[probIndex+2:probIndex+4])
            probIndex += 5
        
        else:
            outputAnsPred[i].append(rawPredictions[probIndex+2])
            probIndex += 4
            
        
    print(outputProbPred)
    print(outputAnsPred)
    print(AnsFromProb)
    
    
    
    
    # Making second data set of rotated numbers to do extra testing
    
    
    reTestArray = np.zeros((800,28,28))
    
    for p, img in enumerate(saved):
        reTestArray[p] = rotation(img, 10)
        #cv2.imshow("rotated",reTestArray[p])
        #cv2.waitKey()
        
    reTestArray = reTestArray.astype('float32')
    reTestArray = reTestArray.reshape(numArray.shape[0], 28, 28, 1)
    
    
    
    predictions2 = model.predict_on_batch(reTestArray)
    numOfNums = len(realPredictions)
    acc = 0.0
    tot = 0.0
    
    rawPredictions = []
    outputProbPred = [ ]
    outputAnsPred =  [[], [], [], [], []]
    AnsFromProb = []
    probNumsSecondCheck = False
    
    for (i, image) in enumerate(saved):
        #print("Top Prediction: ", np.argmax(predictions2[i]))
        
        rawPredictions.append(np.argmax(predictions2[i]))
        
        
        
        
        tot +=1
        if np.argmax(predictions2[i]) == realPredictions[i%numOfNums]:
            acc +=1
        else:
            print("Top Prediction: ", np.argmax(predictions2[i]))
            #cv2.imshow("Missed",image)
            #cv2.waitKey()
        
        #top3 = np.argpartition(predictions[i], -3)[-3:]
        #print("Top 3 Predictions: ", top3[np.argsort(predictions2[i][top3])])
        
        #print("Top 3 values: ",np.array(predictions[i][top3], dtype='double'))
        if i%(numOfNums-1) == 0 and i!=0:
            print(float(acc/numOfNums))
            TOTALACC += float(acc/numOfNums)
        
        
    
    
    
    
    probIndex = 1
    for i in range(5):
        
        outputProbPred.append(rawPredictions[probIndex:probIndex+2])
        
        AnsFromProb.append(outputProbPred[i][0] + outputProbPred[i][1])
        
        
        if probIndex + 3 < len(rawPredictions):
            if rawPredictions[probIndex + 3] == i + 2:
                probNumsSecondCheck = False
            else:
                probNumsSecondCheck = True
        
        
        if AnsFromProb[i] >= 10 and probNumsSecondCheck :
            outputAnsPred[i].append(rawPredictions[probIndex+2:probIndex+4])
            probIndex += 5
        
        else:
            outputAnsPred[i].append(rawPredictions[probIndex+2])
            probIndex += 4
            
        
    print(outputProbPred)
    print(outputAnsPred)
    print(AnsFromProb)

#print("OVERALL ACC:" ,TOTALACC/21)

