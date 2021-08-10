#Command line arguments: 
#   1: Input video 
#   2: Model 
#   3: Combination mode: 
#       o: Only upscaled
#       s: Side by side
#       <number>: Split by <number> pixels
#   4: Output video amplification

import tensorflow as tf
import sys
from ANR_IO import *

nx = 640
ny = 360
chan = 3
frames = 3 
outPath = '../../Datasets/video/output/'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
   
def getArgs():
    if len(sys.argv) == 2:
        inVid = str(sys.argv[1])
        videoLengthInFrames = 999999
    elif len(sys.argv) == 3:
        inVid = str(sys.argv[1])
        videoLengthInFrames = int(sys.argv[2])
    else:
        inVid = str(sys.argv[1])
        sys.exit('Usage: python ANR.py inputvideo.AVI OR: python ANR.py inputvideo.AVI videoLengthInFrames')

    return inVid, videoLengthInFrames 

inVid, videoLength = getArgs()

amp = 256
num = str(118)
inputVid = loadVideo(inVid, 0, videoLength, 0, 640, 60, 420)
autoencoder = tf.keras.models.load_model('./ANR/Models/' + num +'.h5') 
r, g, b = reshapeAndSplitVideo(inputVid)
vid = processVideo(autoencoder, r, g, b, amp = amp)

# The processed video has 4 less frames because every output frame requires 5 frames as input
inputVid = inputVid[1:-1]

pathToVid = inVid.rsplit( ".", 1)[-2]
filename = pathToVid.rsplit( "/")[-1]
outputPath = './ANR/output/'
saveVideo(vid, outputPath + filename + '_output')
