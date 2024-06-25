"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: poll_controller.py

Title: Poll Controller

Author: Chris Lai (MITLL)

Purpose: Demonstrate high-level functionality of a manual controller scheme for the RACECAR.
Functions pull data from the controller and conditional statements are used inside the
update() function to assign speed and angle values to the RACECAR.

Expected Outcome: When the user runs the script, they are able to control the RACECAR
using the following controller buttons:
- When the right trigger is pressed, the RACECAR drives forward
- When the left trigger is pressed, the RACECAR drives backward
- When the left joystick's x-axis has a value of greater than 0, the RACECAR's wheels turns to the right
- When the left joystick's x-axis has a value of less than 0, the RACECAR's wheels turns to the left
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
global speed
global angle

# Set initial values of speed and angle
speed = 0
angle = 0


########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    rc.drive.set_speed_angle(speed, angle)  # Initialize speed to 0


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle

    # [FORWARD/BACKWARD] Variable speed
    # When the triggers are pressed, first detect which trigger is pressed, determine
    # the value of the trigger, and then scale the value of the trigger to the speed
    # !! Right trigger (forward) has priority over left trigger (backwards)
    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0.1:  # Accounts for dead space
        speed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)  # Range is 0.1 - 1
    elif rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0.1:  # Accounts for dead space
        speed = -1 * rc.controller.get_trigger(rc.controller.Trigger.LEFT)  # Range is 0.1 - 1
    else:
        speed = 0

    # [TURNING RIGHT/LEFT] Variable turning angle
    # When the left joystick's x-axis is moved, map the turning angle to the left joystick
    # Turning angle range: [-1, 1], Joystick range: [-1, 1]
    # Account for edge case (deadzone) at -0.1 < x < 0.1
    (x, y) = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    if x > 0.1 or x < -0.1:
        angle = x
    else:
        angle = 0

    print(f"Speed: {speed} || Angle: {angle}")  # Print current speed + angle to terminal (for debug)
    rc.drive.set_speed_angle(speed, angle)  # Send current speed + angle to the RACECAR


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
