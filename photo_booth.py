from __future__ import print_function
from photo_booth_app import PhotoBoothApp
from imutils.video import VideoStream
import argparse
import time

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True, help="path to output directory to store snapshots")
parser.add_argument("-p", "--picamera", type=int, default=-1, help="whether or not to use Raspberry Pi camera")
args = vars(parser.parse_args())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] warming up camera")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# start the app
if __name__ == "__main__":
    pba = PhotoBoothApp(vs, args["output"])
    pba.root.mainloop()
