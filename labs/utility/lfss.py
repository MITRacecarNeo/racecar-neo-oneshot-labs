"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: lfss.py

Title: Line Follower Safety Stop

Author: Chris Lai (MITLL)

Purpose: Given a set of parameters (HSV Threshold, Angle Sensitivity, Speed Value,
Angle Offset, Safety Stop Setpoint, Speed Sensitivity, LIDAR Angle Adjustment), 
enable the RACECAR to complete the line following challenge with safety stop.
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# CHANGE ME (Parameters)
########################################################################################

H_LOW = 0 # Hue lower value between 0 - 179
H_HIGH = 179 # Hue upper value between 0 - 179
S_LOW = 100 # Saturation lower value between 0 - 255
S_HIGH = 255 # Saturation higher value between 0 - 255
V_LOW = 100 # Value lower value between 0 - 255
V_HIGH = 255 # Value higher value between 0 - 255

A_SENSE = 50 # Angle sensitivity between 0% and 100%
A_OFFSET = 0 # Angle offset scaled from -1 to 1

S_VALUE = 50 # Speed magnitude as a percent from 0% to 100%
S_SENSE = 50 # Speed sensitivity between 0% and 100%

SS_SETPOINT = 50 # Safety Stop Setpoint (in cm) between 0cm to 200cm
LIDAR_ANGLE = 25 # LIDAR window (absolute) in degrees from 0deg to 45deg

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
COLOR_THRESH = ((H_LOW, S_LOW, V_LOW),(H_HIGH, S_HIGH, V_HIGH))
CROP_FLOOR = ((180, 0), (rc.camera.get_height(), rc.camera.get_width()))
MIN_CONTOUR_AREA = 30

global speed, angle 
speed = 0
angle = 0


########################################################################################
# Functions
########################################################################################

# [FUNCTION] Update the contour_center and contour_area each frame and display image - threaded
def update_contour():
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()

    # Crop the image to the floor directly in front of the car
    image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the contours of the saved color
        contours = rc_utils.find_contours(image, COLOR_THRESH[0], COLOR_THRESH[1]) # USER PARAM 1-6

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)

        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(image)

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed, angle
    speed = 0
    angle = 0
    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Print start message
    print(
        ">> RACECAR Neo OneShot Demo - Line Follower Safety Stop\n"
        "\n"
        "Controls:\n"
        "   Right Bumper = release RACECAR failsafe\n"
        "   Right Trigger = start driving RACECAR"
    )

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed, angle

    # Call update contour function
    update_contour()

    # Define a basic p-controller. If contour is not found, keep last angle
    if contour_center is not None:
        setpoint = 160
        error = setpoint - contour_center[1]
        kp = -(2/setpoint) * A_SENSE/100*2 # USER PARAM 7
        angle = kp * error + A_OFFSET # USER PARAM 8
        angle = rc_utils.clamp(angle, -1, 1) # clamp between -1 and 1

    # LIDAR data retrieval & controller
    scan = rc.lidar.get_samples()
    window = (360-LIDAR_ANGLE/2, LIDAR_ANGLE/2) # USER PARAM 12
    loc_angle, distance = rc_utils.get_lidar_closest_point(scan, window)
    dist_error = SS_SETPOINT - distance # USER PARAM
    kp_now = -1/SS_SETPOINT * S_SENSE/100 * 2 # USER PARAM

    # Automatically adjust speed based on error if error > setpoint * 2
    if dist_error > -SS_SETPOINT:
        speed = kp_now * dist_error
        speed = rc_utils.clamp(speed, -S_VALUE/100, S_VALUE/100)
    else:
        speed = S_VALUE/100

    # Drive the RACECAR
    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0.1:
        rc.drive.set_speed_angle(speed, angle)
    else:
        rc.drive.set_speed_angle(0, 0)

    # Print speed and angle
    print(f"Speed: {speed}, Angle: {angle}")

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()