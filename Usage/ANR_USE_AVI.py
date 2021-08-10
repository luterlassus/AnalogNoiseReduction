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
    if str(sys.argv[1]) == 'n':
        inVid = './../../DVR/PICT0039.AVI'
    else:
        inVid = str(sys.argv[1])
    num = str(sys.argv[2])
    if sys.argv[3] == 'o':
        comb = 'o'
    elif sys.argv[3] == 's':
        comb = 's'
    else:
        comb = int(sys.argv[3])

    if len(sys.argv) >= 5:
        amp = int(sys.argv[4])
    else:
        amp = 256
    return inVid, num, comb, amp

inVid, num, comb, amp = getArgs()

inputVid = loadVideo(inVid, 100, 300, 0, 640, 60, 420)
autoencoder = tf.keras.models.load_model('../Models/' + num +'.h5') 
r, g, b = reshapeAndSplitVideo(inputVid)
vid = processVideo(autoencoder, r, g, b, amp = amp)

# The processed video has 4 less frames because every output frame requires 5 frames as input
inputVid = inputVid[1:-1]
if comb == 'o':
    print('Saving video, only upscaled version')
elif comb == 's':
    vid = combineVideoSideBySide(inputVid, vid)
else:
    vid = combineVideoSplit(inputVid, vid, comb)

saveVideo(vid, outPath + num)
