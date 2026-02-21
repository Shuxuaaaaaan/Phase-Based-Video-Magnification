import argparse
import sys
import numpy as np
import cv2
import os

from perceptual.filterbank import Steerable
from pyr2arr import Pyramid2arr
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter

# determine what OpenCV version we are using
try:
    import cv2.cv as cv
    USE_CV2 = True
except ImportError:
    # OpenCV 3.x does not have cv2.cv submodule
    USE_CV2 = False


def phaseBasedMagnify(vidFname, vidFnameOut, maxFrames, windowSize, factor, fpsForBandPass, lowFreq, highFreq):

    # initialize the steerable complex pyramid
    steer = Steerable(5)
    pyArr = Pyramid2arr(steer)

    print("Reading:", vidFname, end=" ")

    # get vid properties
    vidReader = cv2.VideoCapture(vidFname)
    if USE_CV2:
        # OpenCV 2.x interface
        vidFrames = int(vidReader.get(cv.CV_CAP_PROP_FRAME_COUNT))    
        width = int(vidReader.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv.CV_CAP_PROP_FPS))
        func_fourcc = cv.CV_FOURCC
    else:
        # OpenCV 3.x interface
        vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))    
        width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vidReader.get(cv2.CAP_PROP_FPS))
        func_fourcc = cv2.VideoWriter_fourcc

    if np.isnan(fps):
        fps = 30

    print(' %d frames' % vidFrames, end="")
    print(' (%d x %d)' % (width, height), end="")
    print(' FPS:%d' % fps)

    # video Writer
    fourcc = func_fourcc(*'MJPG')
    vidWriter = cv2.VideoWriter(vidFnameOut, fourcc, int(fps), (width,height), 1)
    print('Writing:', vidFnameOut)

    # how many frames
    nrFrames = min(vidFrames, maxFrames)

    # setup temporal filter
    filter = IdealFilterWindowed(windowSize, lowFreq, highFreq, fps=fpsForBandPass, outfun=lambda x: x[0])
    #filter = ButterBandpassFilter(1, lowFreq, highFreq, fps=fpsForBandPass)

    print('FrameNr:', end=" ")
    for frameNr in range( nrFrames + windowSize ):
        print(frameNr, end=" ")
        sys.stdout.flush() 

        if frameNr < nrFrames:
            # read frame
            _, im = vidReader.read()
               
            if im is None:
                # if unexpected, quit
                break
            
            # convert to gray image
            if len(im.shape) > 2:
                grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            else:
                # already a grayscale image?
                grayIm = im

            # get coeffs for pyramid
            coeff = steer.buildSCFpyr(grayIm)

            # add image pyramid to video array
            # NOTE: on first frame, this will init rotating array to store the pyramid coeffs                 
            arr = pyArr.p2a(coeff)

            phases = np.angle(arr)

            # add to temporal filter
            filter.update([phases])

            # try to get filtered output to continue            
            try:
                filteredPhases = filter.next()
            except StopIteration:
                continue

            print('*', end=" ")
            
            # motion magnification
            magnifiedPhases = (phases - filteredPhases) + filteredPhases*factor
            
            # create new array
            newArr = np.abs(arr) * np.exp(magnifiedPhases * 1j)

            # create pyramid coeffs     
            newCoeff = pyArr.a2p(newArr)
            
            # reconstruct pyramid
            out = steer.reconSCFpyr(newCoeff)

            # clip values out of range
            out[out>255] = 255
            out[out<0] = 0
            
            # make a RGB image
            rgbIm = np.empty( (out.shape[0], out.shape[1], 3 ) )
            rgbIm[:,:,0] = out
            rgbIm[:,:,1] = out
            rgbIm[:,:,2] = out
            
            #write to disk
            res = cv2.convertScaleAbs(rgbIm)
            vidWriter.write(res.astype(np.uint8))

    # free the video reader/writer
    vidReader.release()
    vidWriter.release()
    print("\n[Done] Outputs saved to", vidFnameOut)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase-Based Video Motion Magnification (EVM Python)")
    parser.add_argument('-v', '--video_path', type=str, required=True, help="Input video path")
    parser.add_argument('-s', '--saving_path', type=str, required=True, help="Output sequence path (.mp4, .avi)")
    parser.add_argument('-a', '--alpha', type=float, default=20.0, help="Amplification factor")
    parser.add_argument('-lo', '--low_omega', type=float, default=72.0, help="Min frequency (hz)")
    parser.add_argument('-ho', '--high_omega', type=float, default=92.0, help="Max frequency (hz)")
    parser.add_argument('-ws', '--window_size', type=int, default=30, help="Sliding window size")
    parser.add_argument('-mf', '--max_frames', type=int, default=60000, help="Max frames to process")
    parser.add_argument('-f', '--fps', type=int, default=600, help="FPS for bandpass. -1 for video native FPS.")
    parser.add_argument('-t', '--threads', type=int, default=1, help="CPU workers (Reserved for future parallel CPU mode)")
    parser.add_argument('-acc', '--accel', choices=['cpu', 'cuda'], default='cpu', help="Acceleration: cpu or cuda (Reserved for future GPU mode)")
    
    args = parser.parse_args()
    
    # Ensure parents directories of saving_path exists
    os.makedirs(os.path.dirname(os.path.abspath(args.saving_path)), exist_ok=True)
    
    phaseBasedMagnify(
        vidFname=args.video_path,
        vidFnameOut=args.saving_path,
        maxFrames=args.max_frames,
        windowSize=args.window_size,
        factor=args.alpha,
        fpsForBandPass=args.fps,
        lowFreq=args.low_omega,
        highFreq=args.high_omega
    )
