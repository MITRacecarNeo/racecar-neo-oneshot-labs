"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: lab3.py

Title: Lab 2 - Wall Follower

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: Write a script to enable fully autonomous behavior from the RACECAR. The
RACECAR should automatically identify the color of a line it sees, then drive on the
center of the line throughout the obstacle course. The RACECAR should also identify
color changes, following colors with higher priority than others. Complete the lines 
of code under the #TODO indicators to complete the lab.

Expected Outcome: When the user runs the script, they are able to control the RACECAR
using the following keys:
- When the right trigger is pressed, the RACECAR moves forward at full speed
- When the left trigger is pressed, the RACECAR, moves backwards at full speed
- The angle of the RACECAR should only be controlled by the center of the line contour
- The RACECAR sees the line color RED as the highest priority, then GREEN, then BLUE
"""


########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import enum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Have the car begin at a stop
    rc.drive.stop()

# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed 
def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global cur_mode

    scan = rc.lidar.get_samples()

    # Reset speed and angle after every frame
    speed = 0
    angle = 0

    # TODO Part 1: 

    rc.drive.set_speed_angle(speed, angle)

    ########################################
    ########### PRINT STATEMENTS ###########
    ########################################

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print calculated statistics when the B button is held down
    if rc.controller.is_down(rc.controller.Button.B):
        print(
            "front_dist {:.2f}, left_dist {:.2f} cm, left_dif {:.2f} cm, right_dist {:.2f} cm, right_dif {:.2f} cm".format(
                front_dist, left_dist, left_dif, right_dist, right_dif
            )
        )

    # Print the current mode when the X button is held down
    if rc.controller.is_down(rc.controller.Button.X):
        print("Mode:", cur_mode)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()