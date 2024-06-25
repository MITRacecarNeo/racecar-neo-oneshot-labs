"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: dot_matrix_test.py

Title: Dot Matrix Test

Author: Chris Lai (MITLL)

Purpose: To test dot matrix functionalities for rapidly displaying debug messages

Expected Outcome: When the buttons are pressed, display the following messages on the
dot matrix display:
- When the A button is pressed, toggle all the lights on and off
- When the B button is pressed, shift a random number between 0-9 to the screen [UNFINISHED]
- When the X button is pressed, show the LIDAR distance to the screen [UNFINISHED]
- When the Y button is pressed, write some text on the screen using the scroll feature [UNFINISHED]
"""

########################################################################################
# Imports
########################################################################################

import sys
import numpy as np

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(0, '../library')
import racecar_core

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
global mat

# Create new empty matrix as a display
mat = rc.display.new_matrix()


########################################################################################
# Functions
########################################################################################

# [FUNCTION] Returns a number 0-9 in an 8x8 matrix
def disp_num(num):
    arr = np.empty(shape=(8, 8))
    arr.fill(0)  # Init numpy array of size 8 (rows) x 8 (columns) with all zeros

    if num < 0 or num > 9:
        print(f"[disp_num] Error: Number out of range! Enter a number between 0-9.")
        return None
    else:
        if num == 0:
            arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]])

    return arr


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    pass  # Remove 'pass' and write your source code for the start() function here


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    pass  # Remove 'pass' and write your source code for the update() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
