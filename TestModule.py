'''
Created on Aug 21, 2021

@author: Mrmon
'''
# -*- encoding: UTF-8 -*-

# This is just an example script that shows how images can be accessed
# through ALVideoDevice in python.



import sys
import time

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



if __name__ == '__main__':
  IP = "169.254.213.151"  # Replace here with your NaoQi's IP address.
  PORT = 9559

  # Read IP address from first argument if any.
  if len(sys.argv) > 1:
    IP = sys.argv[1]

  naoImage = showNaoImage(IP, PORT)
  textProxy = ALProxy("ALTextToSpeech", IP, PORT)
  textProxy.setParameter("doubleVoiceTimeShift",0)
  textProxy.say("Hello! How are you doing?")
  
