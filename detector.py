#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Detect objects on a LIVE camera feed using
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import cv2
import sys
import numpy
import ntpath
import argparse
import picamera
import picamera.array

import mvnc.mvncapi as mvnc

from utils import visualize_output
from utils import deserialize_output

import RPi.GPIO as GPIO
import threading
import time

# Stepper motor
control_pins=[7,11,13,15]
halfstep_seq = [
  [1,0,0,0],
  [1,1,0,0],
  [0,1,0,0],
  [0,1,1,0],
  [0,0,1,0],
  [0,0,1,1],
  [0,0,0,1],
  [1,0,0,1],
]


# Detection threshold: Minimum confidance to tag as valid detection
CONFIDANCE_THRESHOLD = 0.60 # 60% confidant

# Variable to store commandline arguments
ARGS                 = None

# OpenCV object for video capture
# camera               = None

# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph( blob )

    return graph

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image( frame ):

    # Resize image [Image size is defined by choosen network, during training]
    img = cv2.resize( frame, tuple( ARGS.dim ) )

    # Convert RGB to BGR [OpenCV reads image in BGR, some networks may need RGB]
    if( ARGS.colormode == "rgb" ):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img, frame, motor ):

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img, 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # Deserialize the output into a python dictionary
    output_dict = deserialize_output.ssd( 
                      output, 
                      CONFIDANCE_THRESHOLD, 
                      frame.shape )

    # Print the results (each image/frame may have multiple objects)
    print( "I found these objects in "
            + " ( %.2f ms ):" % ( numpy.sum( inference_time ) ) )

    count_person = 0
    for i in range( 0, output_dict['num_detections'] ):
        # Only interested in person.
        if labels[output_dict['detection_classes_' + str(i)]] != "15: person":
            continue
        if count_person > 0:
            continue
        count_person = count_person + 1
        print( "%3.1f%%\t" % output_dict['detection_scores_' + str(i)] 
               + labels[ int(output_dict['detection_classes_' + str(i)]) ]
               + ": Top Left: " + str( output_dict['detection_boxes_' + str(i)][0] )
               + " Bottom Right: " + str( output_dict['detection_boxes_' + str(i)][1] ) )

        # Draw bounding boxes around valid detections 
        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]
        print("x1 x2: %d %d\n" % (x1, x2))
        mid = (x1+x2)/2
        dis = 320 - mid
        print("distance: %d\n" % dis)
        if dis > 20:
            motor.direction = "forward"
        elif dis < -20:
            motor.direction = "backward"
        else:
            motor.direction = "stop"

        # Prep string to overlay on the image
        display_str = ( 
                labels[output_dict.get('detection_classes_' + str(i))]
                + ": "
                + str( output_dict.get('detection_scores_' + str(i) ) )
                + "%" )

        frame = visualize_output.draw_bounding_box( 
                       y1, x1, y2, x2, 
                       frame,
                       thickness=4,
                       color=(255, 255, 0),
                       display_str=display_str )
    print( '\n' )

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        cv2.imshow( 'NCS live inference', frame )

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()
    cv2.destroyAllWindows()

class Motor(object):
    def __init__(self, direction = "stop"):
        self.direction = direction
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        while True:
            if self.direction == "stop":
                time.sleep(0.3)
            elif self.direction == "forward":
                for halfstep in range(8):
                    for pin in range(4):
                        GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
                    time.sleep(0.001)
            elif self.direction == "backward":
                for halfstep in reversed(range(8)):
                    for pin in range(4):
                        GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
                    time.sleep(0.001)

class Camera(object):
    def __init__(self):
        self.frame = ''
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start() 

    def run(self):
        with picamera.PiCamera() as camera:
            with picamera.array.PiRGBArray(camera) as frame:
                while( True ):
                    camera.resolution = ( 640, 480 )
                    camera.capture( frame, ARGS.colormode, use_video_port=True )
                    self.frame = frame.array
                    frame.seek( 0 )
                    frame.truncate()


# ---- Main function (entry point for this script ) --------------------------

def main(motor):

    device = open_ncs_device()
    graph = load_graph( device )
    cam = Camera() 
    time.sleep(1)

    # Main loop: Capture live stream & send frames to NCS
    while( True ):
        frame = cam.frame
        img = pre_process_image( frame )
        infer_image( graph, img, frame, motor )
            

        # Display the frame for 5ms, and close the window so that the next
        # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
        if( cv2.waitKey( 5 ) & 0xFF == ord( 'q' ) ):
            break

    close_ncs_device( device, graph )

# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Detect objects on a LIVE camera feed using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/SSD_MobileNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-v', '--video', type=int,
                         default=0,
                         help="Index of your computer's V4L2 video device. \
                               ex. 0 for /dev/video0" )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../caffe/SSD_MobileNet/labels.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[127.5, 127.5, 127.5],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=0.00789,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[300, 300],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="bgr",
                         help="RGB vs BGR color sequence. This is network dependent." )

    ARGS = parser.parse_args()

    # Create a VideoCapture object
    # camera = cv2.VideoCapture( ARGS.video )

    # Set camera resolution
    # camera.set( cv2.CAP_PROP_FRAME_WIDTH, 620 )
    # camera.set( cv2.CAP_PROP_FRAME_HEIGHT, 480 )

    # Load the labels file
    labels =[ line.rstrip('\n') for line in
              open( ARGS.labels ) if line != 'classes\n']
    
    # Setup GPIO pins.
    GPIO.setmode(GPIO.BOARD)
    for pin in control_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, 0)

    motor = Motor()

    main(motor)
    GPIO.cleanup()

# ==== End of file ===========================================================
