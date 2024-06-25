"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: safety_stop.py

Title: Safety Stop

Author: Chris Lai (MITLL)

Purpose: Prevent the RACECAR from hitting objects in the front or back side using
the LIDAR and a safety stop paradigm. Use buttons to adjust the absolute speed of the car
and distance threshold to create a safe and manageable driving environment.

Expected Outcome: When the user runs the script, they are able to control the RACECAR
using the following controller buttons:
- When the right trigger is pressed, the RACECAR drives forward
- When the left trigger is pressed, the RACECAR drives backward
- When the A button is pressed, increase the speed of the RACECAR
- When the B button is pressed, decrease the speed of the RACECAR
- When the X button is pressed, increase the distance threshold the RACECAR should stop at
- When the Y button is pressed, decrease the distance threshold the RACECAR should stop at
"""

########################################################################################
# Imports
########################################################################################

import sys

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
global speed
global speed_div
global dist_thresh

# Declare static variables - for safety reasons the user should not be allowed to go under
# 30cm or above 100cm when adjusting the values
DIST_UPPER = 100
DIST_LOWER = 30

# Set initial values for global varaibles
speed = 0
speed_div = 5
dist_thresh = 30


########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    rc.drive.set_speed_angle(speed, 0)  # Angle will always be set to 0 for this lab


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global speed_div
    global dist_thresh

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

    # [SPEED MOD] If A or B buttons are pressed, increase or decrease overall speed
    if rc.controller.was_pressed(rc.controller.Button.A):
        if speed_div > 1.1:
            speed_div -= 0.25  # Increase speed by reducing denominator
        else:
            speed_div = 1.0
        print(f"Current Speed Modifier: speed * 1/{speed_div}")
    if rc.controller.was_pressed(rc.controller.Button.B):
        if speed_div < 9.9:
            speed_div += 0.25  # Decrease speed by increasing denominator
        else:
            speed_div = 10.0
        print(f"Current Speed Modifier: speed * 1/{speed_div}")

    # [DIST THRESH MOD] If X or Y buttons are pressed, increase or decrease distance threshold
    if rc.controller.was_pressed(rc.controller.Button.X):
        if dist_thresh < DIST_UPPER - 0.5:  # 0.5 is for tolerance
            dist_thresh += 1  # Increase distance threshold by 1cm
        else:
            dist_thresh = DIST_UPPER
        print(f"Current Distance Threshold: {dist_thresh}cm")
    if rc.controller.was_pressed(rc.controller.Button.Y):
        if dist_thresh > DIST_LOWER + 0.5:  # 0.5 is for tolerance
            dist_thresh -= 1  # Decrease distance threshold by 1cm
        else:
            dist_thresh = DIST_LOWER
        print(f"Current Distance Threshold: {dist_thresh}cm")

    # [SAFETY STOP] If LIDAR detects an object in front or behind the car within the distance threshold,
    # override the speed and stop the car.
    scan = rc.lidar.get_samples()  # Return 505 samples in an array, CCW
    fw_dist = rc_utils.get_lidar_average_distance(scan, 180)  # 180 deg is in front of the car
    bw_dist = rc_utils.get_lidar_average_distance(scan, 0)  # 0 deg is behind the car

    # If forward distance < threshold and right trigger is pressed, stop the car
    if fw_dist < dist_thresh and rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0.1:
        speed = 0
        print(f"[VIOLATION]: fw_dist > dist_thresh || fw_dist: {fw_dist}, dist_thresh: {dist_thresh}")

    # If backwards distance < threshold and left trigger is pressed, stop the car
    if bw_dist < dist_thresh and rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0.1:
        speed = 0
        print(f"[VIOLATION]: bw_dist > dist_thresh || bw_dist: {bw_dist}, dist_thresh: {dist_thresh}")

    # Calculate adjusted speed and send to car
    real_speed = speed * 1.0 / speed_div
    print(f"Real Speed: {real_speed}")
    rc.drive.set_speed_angle(real_speed, 0)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
