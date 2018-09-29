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

import wiringpi

# motor
control_pins=[7,0,2,3]


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
    numpy.subtract(img, numpy.float16( ARGS.mean ),out=img)
    img = img * ARGS.scale
    #img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img, frame, motor, pid ):

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
    # print( "I found these objects in "
    #        + " ( %.2f ms ):" % ( numpy.sum( inference_time ) ) )

    count_person = 0
    for i in range( 0, output_dict['num_detections'] ):
        # Only interested in person.
        if labels[output_dict['detection_classes_' + str(i)]] != "15: person":
            continue
        if count_person > 0:
            continue
        count_person = count_person + 1
        #print( "%3.1f%%\t" % output_dict['detection_scores_' + str(i)]
        #       + labels[ int(output_dict['detection_classes_' + str(i)]) ]
        #       + ": Top Left: " + str( output_dict['detection_boxes_' + str(i)][0] )
        #       + " Bottom Right: " + str( output_dict['detection_boxes_' + str(i)][1] ) )

        # Draw bounding boxes around valid detections
        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]
        mid = (x1+x2)/2
        dis = 160 - mid
        pid.update(dis)
        print("distance: %d " % dis, end='', flush=True)
        if pid.output > 30:
            motor.direction = int(numpy.interp(pid.output,[0,160],[1,100]))
            print("direction: %d\n" % motor.direction)
        elif pid.output < -30:
            motor.direction = int(numpy.interp(pid.output,[-160,0],[-100,-1]))
            print("direction: %d\n" % motor.direction)
        else:
            motor.direction = 0

        # Prep string to overlay on the image
        #display_str = (
        #        labels[output_dict.get('detection_classes_' + str(i))]
        #        + ": "
        #        + str( output_dict.get('detection_scores_' + str(i) ) )
        #        + "%" )

        #frame = visualize_output.draw_bounding_box(
        #               y1, x1, y2, x2,
        #               frame,
        #               thickness=4,
        #               color=(255, 255, 0),
        #               display_str=display_str )

    if count_person == 0:
        motor.direction = 0

    # If a display is available, show the image on which inference was performed
    #if 'DISPLAY' in os.environ:
        #cv2.imshow( 'NCS live inference', frame )

# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()
    cv2.destroyAllWindows()

class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

class Motor(object):
    def __init__(self, direction = 0):
        self.direction = 0
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        last_direction = 0
        while True:
            if self.direction == last_direction:
               time.sleep(0.02)
               continue
            else:
               last_direction = self.direction
            if self.direction == 0:
                wiringpi.softPwmWrite(7,0)
                wiringpi.softPwmWrite(0,0)
                wiringpi.softPwmWrite(2,0)
                wiringpi.softPwmWrite(3,0)
            elif self.direction > 0:
                wiringpi.softPwmWrite(3,0)
                wiringpi.softPwmWrite(0,0)
                wiringpi.softPwmWrite(7,self.direction)
                wiringpi.softPwmWrite(2,self.direction)
            elif self.direction < 0:
                wiringpi.softPwmWrite(7,0)
                wiringpi.softPwmWrite(2,0)
                wiringpi.softPwmWrite(0,-self.direction)
                wiringpi.softPwmWrite(3,-self.direction)

class Camera(object):
    def __init__(self):
        self.frame = ''
        self.img = ''
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        with picamera.PiCamera(resolution=(320,240), framerate=90) as camera:
            with picamera.array.PiRGBArray(camera) as frame:
                while( True ):
                    camera.capture( frame, ARGS.colormode, use_video_port=True )
                    self.frame = frame.array
                    frame.seek( 0 )
                    frame.truncate()

class Process(object):
    def __init__(self, frame):
        self.frame = frame
        self.img = ''
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        while(True):
            self.img = pre_process_image(self.frame)



# ---- Main function (entry point for this script ) --------------------------

def main(motor, pid):

    device = open_ncs_device()
    graph = load_graph( device )
    cam = Camera()
    time.sleep(1)
    pre = Process(cam.frame)
    time.sleep(1)

    # Main loop: Capture live stream & send frames to NCS
    while( True ):
        frame = cam.frame
        pre.frame = frame
        img = pre.img
        infer_image( graph, img, frame, motor, pid )


        # Display the frame for 5ms, and close the window so that the next
        # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
        #if( cv2.waitKey( 5 ) & 0xFF == ord( 'q' ) ):
        #    break

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
    wiringpi.wiringPiSetup()
    for pin in control_pins:
        wiringpi.pinMode(pin,1)
        wiringpi.softPwmCreate(pin,0,100)

    motor = Motor()
    pid = PID(1.3,0,0.3)
    pid.SetPoint=0.0
    pid.setSampleTime(0.1)

    main(motor, pid)

# ==== End of file ===========================================================
