'''
Created on Aug 21, 2021

Unrefined testing for first iteration
'''
# -*- encoding: UTF-8 -*-

# This is just an example script that shows how images can be accessed
# through ALVideoDevice in python.



import sys
import time
import math

# Python Image Library
from PIL import Image

from naoqi import ALProxy


def showNaoImage(IP, PORT):
  """
  First get an image from Nao, then show it on the screen with PIL.
  """

  camProxy = ALProxy("ALVideoDevice", IP, PORT)
  resolution = 2    # VGA 2
  colorSpace = 11    # RGB 11  10 might be black and white
  
  # width, height, num of layers, colorspace, time stamp from qi::clock seconds, microseconds, binary array of size height*width*nblayers with image data, cam ID 0 or 1, left angle(radian). top angle, right angle ,bottom angle
  Indexes = []
  videoClient = camProxy.subscribeCamera("python_client",0 ,resolution, colorSpace, 5)

  t0 = time.time()

  # Get a camera image.
  # image[6] contains the image data passed as an array of ASCII chars.
  naoImage = camProxy.getImageRemote(videoClient)

  t1 = time.time()

  # Time the image transfer.
  print "acquisition delay ", t1 - t0

  camProxy.unsubscribe(videoClient)


  # Now we work with the image returned and save it as a PNG  using ImageDraw
  # package.

  # Get the image size and pixel array.
  imageWidth = naoImage[0]
  imageHeight = naoImage[1]
  array = naoImage[6]

  # Create a PIL Image from our pixel array.
  im = Image.frombytes("RGB", (imageWidth, imageHeight), array)
  # changed to bytes
  # Save the image.
  im.save("camImage.png", "PNG")

  im.show()


def StiffnessOn(proxy):
    pNames = "Body"
    pStiffnessLists = 0.5
    pTimeLists = 0.5
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)
    

if __name__ == '__main__':
  IP = "169.254.69.241"  # Replace here with your NaoQi's IP address.
  PORT = 9559

  # Read IP address from first argument if any.
  if len(sys.argv) > 1:
    IP = sys.argv[1]

  
  # Create a proxy to ALLandMarkDetection
  markProxy = ALProxy("ALLandMarkDetection", IP, PORT)
  # Subscribe to the ALLandMarkDetection extractor
  period = 500 
  markProxy.subscribe("Test_Mark", period, 0.0 )
  
  memProxy = ALProxy("ALMemory", IP, PORT)
  # Get data from landmark detection (assuming face detection has been activated).
  
  # Fix headyaw for image capture
  motionProxy = ALProxy("ALMotion", IP,PORT)
  
  postureProxy = ALProxy("ALRobotPosture", IP, PORT)

  # Set NAO in Stiffness On
  StiffnessOn(motionProxy)

  # Send NAO to Pose Init
  postureProxy.goToPosture("Crouch", 0.5)
  
  effectorName = "Head"

  # Active Head tracking
  isEnabled    = True
  motionProxy.wbEnableEffectorControl(effectorName, isEnabled)

    # Example showing how to set orientation target for Head tracking
    # The 3 coordinates are absolute head orientation in NAO_SPACE
    # Rotation in RAD in x, y and z axis

    # X Axis Head Orientation feasible movement = [-20.0, +20.0] degree
    # Y Axis Head Orientation feasible movement = [-75.0, +70.0] degree
    # Z Axis Head Orientation feasible movement = [-30.0, +30.0] degree

  targetCoordinateList = [
    [00.0,  00.0,  00.0], # target 0
    [00.0,  00.0,  00.0] # target 1
   # [00.0,  00.0,  00.0], # target 2
   # [00.0,  00.0,  00.0], # target 3
   # [00.0,  00.0,  00.0], # target 4
   # [00.0,  00.0,  00.0], # target 5
   # [00.0,  00.0,  00.0], # target 6
   # [00.0,  00.0,  00.0], # target 7
   # [00.0,  00.0,  00.0], # target 8
    ]

    # wbSetEffectorControl is a non blocking function
    # time.sleep allow head go to his target
    # The recommended minimum period between two successives set commands is
    # 0.2 s.
  for targetCoordinate in targetCoordinateList:
      targetCoordinate = [target*math.pi/180.0 for target in targetCoordinate]
      motionProxy.wbSetEffectorControl(effectorName, targetCoordinate)
      time.sleep(3.0)

  
  
  allSee = 0
  
  
  markArray = [0,0,0,0]
  
  none = []
  for i in range(21):
      textProxy = ALProxy("ALTextToSpeech", IP, PORT)
      textProxy.setParameter("doubleVoiceTimeShift",0)
      textProxy.say("test")
      data = memProxy.getData("LandmarkDetected")
      markArray = [0,0,0,0]
      #print(data)
      
      #Getting marks
      try:
          for x in range(len(data[1])):
              markArray[x] = data[1][x][1][0]
          markArray.sort() # sorting marks
                        
        
          for i in range(4):
              if markArray[i] > 0:
                  markArray[i] = 1
        
          yMove = [markArray[0] | markArray[1], markArray[2] | markArray[3]] # [0,1] Move up       # [1,0] Move Down
          zMove = [markArray[0] | markArray[2], markArray[1] | markArray[3]] # [0, 1] Move left    # [1, 0] Move Right
                   # see left                       see right
          print(" Y then Z")
          print(yMove)
          print(zMove)
          
          targetCoordinateList = [[00.0,  yMove[0]*20.00 - yMove[1]*20.0, -zMove[0]*20.00 + zMove[1]*20.0]]
      
      
          for targetCoordinate in targetCoordinateList:
              targetCoordinate = [target*math.pi/180.0 for target in targetCoordinate]
              motionProxy.wbSetEffectorControl(effectorName, targetCoordinate)
              time.sleep(5.0)
              
      except Exception, e:
          print("Not all seen...")
          
      
      

      try:
          allSee = data[1][0][1][0] + data[1][1][1][0] + data[1][2][1][0] + data[1][3][1][0]
          
      except Exception, e:
          print("Not all seen...")
          
          
      
      if allSee == 296:
          naoImage = showNaoImage(IP, PORT)
          allSee = 0


    # Deactivate Head tracking
  postureProxy.goToPosture("Stand", 0.5)
  isEnabled    = False
  motionProxy.wbEnableEffectorControl(effectorName, isEnabled)


  
  
  
