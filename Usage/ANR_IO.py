#Funcions for input, output and reshaping
import cv2
import numpy as np
from skimage.transform import resize

def loadVideo(filename, startFrame = 0, length = 9999, x = 0, w = 640, y = 60, h = 420):
    """ Loads the video stored at filename. Specify startframe and length to save time and ram """
    #Open the video file using cv2
    cap = cv2.VideoCapture(filename)
    
    #Getting the video dimentions
    stopFrame = startFrame + length
    readVidLen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if readVidLen < stopFrame:
        print("Video: ", filename, "is shorter than", stopFrame, "frames, loading all", readVidLen, "frames")
        stopFrame = readVidLen - 35
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     
    #Generate a buffer to fill with the video
    buf = np.empty((stopFrame, frameHeight, frameWidth, 3), np.dtype('uint8'))

    #Load the video into memory frame by frame
    fc = 0
    ret = True
    while (fc < stopFrame and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    #Crop and return the video
    return buf[startFrame:stopFrame, y:h, x:w, :]

def saveVideo(video, outputName,fps = 30):
    """ Saves the video as outputname.AVI """
    pathout = outputName +'.AVI'
    print(video.shape)
    size = (video.shape[2],video.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    out = cv2.VideoWriter(pathout,  fourcc , fps, size)
    for i in range(len(video)):
        # writing to a image array
        out.write(video[i])
    out.release()
    
#Functions for reshaping the input video
def getFramesAround(DS, num, fbf = 1, faf = 1):
     """ Returns a matrix with the fbf frames before and the faf frames after frame number num """
     return np.stack(DS[(num-fbf):(1+num+faf)], axis = 2)

def reshapeDStoFBF(X, fbf = 1, faf = 1):
     """ Reshapes the dataset so that each entry contains the fbf frames before and faf frames after the entry """
     rn = fbf
     buf =  np.empty((len(X) - (faf + fbf), len(X[0]), len(X[0][0]), fbf + faf + 1), np.dtype('uint8'))
     for i in range(len(X) - (faf + fbf)):
         buf[rn - fbf] = getFramesAround(X, rn,fbf,faf)
         rn += 1
     print('Reshaped dataset to size: ', buf.shape)
     return buf

def reshapeAndSplitVideo(video):
     """ Reshapes the video to a dataset with 5 frames per instance, and splits it into each color """
     r = reshapeDStoFBF(video[:, :, :, 0])
     g = reshapeDStoFBF(video[:, :, :, 1])
     b = reshapeDStoFBF(video[:, :, :, 2])
     return r, g, b
          

#Functions for combining the input and output video
def combineVideoSplit(video1, video2, ratio):
     """ Combines video1 and video2 using horizontal split ratio in pixels """
     combinedVid = np.append(video1[:, :, :ratio], video2[:, :, ratio:], axis = 2)
     return combinedVid

def combineVideoSideBySide(video1, video2):
     """ Combines video1 and video2 side by side """
     combinedVid = np.append(video1[:, :, :], video2[:, :, :], axis = 2)
     return combinedVid

#Function for using the model
def processVideo(model, r, g, b, amp = 256):
     """ Processes the video chanel by chanel using the model provided """
     dr = np.clip(model.predict(r.astype(float) / 255.0, batch_size = 1), 0.0, 1.0)
     dg = np.clip(model.predict(g.astype(float) / 255.0, batch_size = 1), 0.0, 1.0)
     db = np.clip(model.predict(b.astype(float) / 255.0, batch_size = 1), 0.0, 1.0)
     print(dr.shape, dg.shape, db.shape)
     video = np.append(dr, dg, axis = 3)
     video = np.append(video, db, axis = 3)
     video = (video * amp).astype(np.uint8)
     return video
                     
