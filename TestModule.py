'''
Testing program for Nao Senior design Worksheet Capture module
'''
# -*- encoding: UTF-8 -*-

# This is just an example script that shows how images can be accessed
# through ALVideoDevice in python.
# name of project as prefix to function names


import sys
import time
import math

# Python Image Library
from PIL import Image

from naoqi import ALProxy
class var:
    K = 0


def showNaoImage(IP, PORT):
  """
  First get an image from Nao, then show it on the screen with PIL.
  """

  camProxy = ALProxy("ALVideoDevice", IP, PORT)
  resolution = 2    # VGA 2
  colorSpace = 11   # RGB 11  10 is negative I think
  
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
  # Save the image.\
  image_name= "NaoPage1TestImage" + str(var.K) + ".png"
  im.save(image_name, "PNG")
  var.K+=1

  im.show()
  return im


def StiffnessOn(proxy):
    pNames = "Body"
    pStiffnessLists = 0.5
    pTimeLists = 0.5
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)
    


#New
def StartMarkDetection(proxy, period):
     
    proxy.subscribe("Test_Mark", period, 0.0 )
    
def SimplePaperLocation(markData):
    #markArray = [1,1,1,1]
    thisdict = {
             64 : " upper left, ",
             68 : " upper right, ",
             80 : " lower left, ",
             84 : " lower right, "
            }
      
    #Getting marks 
    # it was try
    try:
        #get mark data
        for x in range(len(markData[1])):
            thisdict[markData[1][x][1][0]] = ""
            
            
        
        #markArray.sort() # sorting marks
                        
        
        #for i in range(4):
            #if markArray[i] > 1:
                #markArray[i] = 0 #see it
            #else:
                #markArray[i] = 1 #don't see it
        
        papDirection = thisdict[64] + thisdict[68] + thisdict[80] + thisdict[84]
        #if (markArray[0]):
            #papDirection += " upper left, "
        #if (markArray[1]):
            #papDirection += " upper right, "
        #if (markArray[2]):
            #papDirection += " lower left, "
        #if (markArray[3]):
            #papDirection += " lower right, "
        
        #yMove = [markArray[0] | markArray[1], markArray[2] | markArray[3]] # [0,1] Move up       # [1,0] Move Down
        #zMove = [markArray[0] | markArray[2], markArray[1] | markArray[3]] # [0, 1] Move left    # [1, 0] Move Right
                   # see left                       see right
          
        #targetCoordinateList = [[00.0,  yMove[0]*20.00 - yMove[1]*20.0, -zMove[0]*20.00 + zMove[1]*20.0]]
      
      
        #for targetCoordinate in targetCoordinateList:
            #targetCoordinate = [target*math.pi/180.0 for target in targetCoordinate]
            #motionProxy.wbSetEffectorControl(effectorName, targetCoordinate)
            #time.sleep(5.0)
              
    except Exception, e:
    #else:
        print("Not seen at all... Sorry!")
    
    papDirection = thisdict[64] + thisdict[68] + thisdict[80] + thisdict[84]
    return papDirection





if __name__ == '__main__':
  IP = "169.254.195.214"  # Replace here with your NaoQi's IP address.
  PORT = 9559

  # Read IP address from first argument if any.
  if len(sys.argv) > 1:
    IP = sys.argv[1]


  # Subscribe to the ALLandMarkDetection extractor
  StartMarkDetection(ALProxy("ALLandMarkDetection", IP, PORT), 50)
  
  # Subscribe to ALTextToSpeech
  textProxy = ALProxy("ALTextToSpeech", IP, PORT)
  
  # Subscribe to ALMemory proxy 
  memProxy = ALProxy("ALMemory", IP, PORT)
  
  # Fix head for image capture
  motionProxy = ALProxy("ALMotion", IP,PORT)
  
  postureProxy = ALProxy("ALRobotPosture", IP, PORT)

  # Set NAO in Stiffness On
  StiffnessOn(motionProxy)

  # Send NAO to Pose crouch
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
    [00.0,  00.0,  00.0]   # target 0
   # [00.0,  00.0,  00.0]  # target 1
   # [00.0,  00.0,  00.0], # target 2
   # [00.0,  00.0,  00.0], # target 3
   # [00.0,  00.0,  00.0], # target 4
   # [00.0,  00.0,  00.0], # target 5
   # [00.0,  00.0,  00.0], # target 6
   # [00.0,  00.0,  00.0], # target 7
   # [00.0,  00.0,  00.0], # target 8
    ]


  for targetCoordinate in targetCoordinateList:
      targetCoordinate = [target*math.pi/180.0 for target in targetCoordinate]
      motionProxy.wbSetEffectorControl(effectorName, targetCoordinate)
      time.sleep(3.0)

  
  
  allSee = 0
  sayCount =0 
  
  textProxy.say("Hold you paper up as instructed on the back, and say cheese for your picture!")
  time.sleep(3.0)
  seen = 0
  
  while(seen < 21):
  #while (allSee != 296):
      #textProxy.say("test")
      
      # Get data from landmark detection (assuming face detection has been activated).
      data = memProxy.getData("LandmarkDetected")
      
      
      
      try:
          allSee = data[1][0][1][0] + data[1][1][1][0] + data[1][2][1][0] + data[1][3][1][0]
          print(sayCount)

      except Exception, e:
          print("Not all seen...")
          sayCount += 1
          #Getting marks location array [TopL, TopR, BotL, BotR]
          if sayCount%1000 == 0:
              notSeen = "I can't see the"
              notSeen += SimplePaperLocation(data) + " Nao landmarks try adjusting!"
              textProxy.say(notSeen)

          
      
      if allSee == 296: #read all the marks on the page
          naoPaper = showNaoImage(IP, PORT)
          allSee = 0
          seen = seen + 1
      


  # Deactivate Head tracking and activate movement
  postureProxy.goToPosture("Stand", 0.5)
  isEnabled    = False
  motionProxy.wbEnableEffectorControl(effectorName, isEnabled)


  
  
  
