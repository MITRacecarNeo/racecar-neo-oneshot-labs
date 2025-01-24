"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: hsv_tuner_non_gui.py

Title: HSV Tuner

Author: Paul Thai

Purpose: User is able to modify the HSV thresholds to create their own filters
and affect the precision of line following in the RACECAR. Codebase uses the terminal
to modify the HSV values
"""

import sys
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import threading

# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../library')
import racecar_core
import racecar_utils as rc_utils

# Create RACECAR object
rc = racecar_core.create_racecar()

# Default Variables
global H_low, H_high, S_low, S_high, V_low, V_high
H_low, H_high = 0, 179
S_low, S_high = 0, 255
V_low, V_high = 0, 255

COLOR_THRESH = ((H_low, S_low, V_low), (H_high, S_high, V_high))
CROP_FLOOR = ((180, 0), (rc.camera.get_height(), rc.camera.get_width()))
speed_div = 7  # How much speed should be divided
angle_div = 3  # How much angle should be divided
mode_mod = [1,2,3,4,5,6] # H_low, H_high, S_low, S_high, V_low, V_high respectively
mode_select = 0
cur_mode = 0

speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

MIN_CONTOUR_AREA = 30

# [FUNCTION] Update the contour_center and contour_area each frame and display image
def update_contour(img):
    global contour_center
    global contour_area
    global tk_image

    # Crop the image to the floor directly in front of the car
    image = rc_utils.crop(img, CROP_FLOOR[0], CROP_FLOOR[1])

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the contours of the saved color
        contours = rc_utils.find_contours(image, COLOR_THRESH[0], COLOR_THRESH[1])

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


# [FUNCTION] Start function isn't really needed here
def start():
    # Set initial driving speed and angle
    rc.drive.set_speed_angle(0, 0)

    # Print start message
    print(
        ">> RACECAR Neo OneShot Demo - Line Follower with Live HSV Tuning\n"
        "\n"
        "Controls:\n"
        "   A button (1) = save tuned HSV value of line to system\n"
        "   B button (2) = change between different HSV modifier modes\n"
        "   X button (3) = increase the value of the selected mode\n"
        "   Y button (4) = decrease the value of the selected mode\n"
        "   LB button (z) = display the values of your HSV\n"
        "   RB button (?) = reset the HSV values to their default values\n"
    )


# [FUNCTION] Tune HSV values to RACECAR camera while in main loop
def update():
    global COLOR_THRESH
    global mode_mod
    global mode_select
    global cur_mode
    global speed_div
    global angle_div

    global speed
    global angle

    global H_low, H_high, S_low, S_high, V_low, V_high

    # Make manual mask for updating colors
    img = rc.camera.get_color_image()
    img = cv.resize(img, (320, 240))
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # Logitech camera returns BGR image
    hsv_low = np.array([H_low, S_low, V_low], np.uint8)
    hsv_high = np.array([H_high, S_high, V_high], np.uint8)
    mask = cv.inRange(hsv_image, hsv_low, hsv_high)
    cv.imshow('mask', mask)

    # Update contour function
    update_contour(img)

    # Choose an angle based on contour_center
    # If we could not find a contour, keep the previous angle
    if contour_center is not None:
        setpoint = 160
        error = setpoint - contour_center[1]
        angle = rc_utils.remap_range(error, -setpoint, setpoint, 1, -1)

    # Modify speed and angle commands by physical RACECAR modifier
    # speed = 1 / speed_div
    # angle /= angle_div
    # Speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Send speed and angle commands to RACECAR 
    rc.drive.set_speed_angle(speed, angle)

    ######################
    # CONTROLLER OPTIONS #
    ######################

    # When A button is pressed, save the HSV Threshold to global variable
    if rc.controller.was_pressed(rc.controller.Button.A):
        print(f"HSV Threshold Saved!: ({H_low}, {S_low}, {V_low}), ({H_high}, {S_high}, {V_high})")
        COLOR_THRESH = ((H_low, S_low, V_low), (H_high, S_high, V_high))

    # When B button is pressed, different HSV modes
    if rc.controller.was_pressed(rc.controller.Button.B):
        # Iterate through the list if button is pressed
        cur_mode = mode_mod[mode_select]

        # Determine which mode we are in
        if cur_mode == 1:
            print(f"Currently Modifying H_low: Value = {H_low}")
        if cur_mode == 2:
            print(f"Currently Modifying S_low: Value = {S_low}" )
        if cur_mode == 3:
            print(f"Currently Modifying V_low: Value = {V_low}")
        if cur_mode == 4:
            print(f"Currently Modifying H_high: Value = {H_high}")
        if cur_mode == 5:
            print(f"Currently Modifying S_high: Value = {S_high}")
        if cur_mode == 6:
            print(f"Currently Modifying V_high: Value = {V_high}")

        # Increment the Count
        mode_select += 1

        # Check to see if out of bounds
        if mode_select > 5:
            # Reset it back to zero
            mode_select = 0


    # When X button is pressed, increase the HSV values
    if rc.controller.is_down(rc.controller.Button.X):
        # Determine which mode we increment
        if cur_mode == 1:
            if H_low > 179:
                print("Cannot increase the value of H_low pass 179")
                H_low = 179
            else:
                H_low += 1
                print(f"Currently increasing H_low: Value = {H_low}")
        elif cur_mode == 2:
            if S_low > 255:
                print("Cannot increase the value of S_low pass 255")
                S_low = 255
            else:
                S_low += 1
                print(f"Currently increasing S_low: Value = {S_low}" )
        elif cur_mode == 3:
            if V_low > 255:
                print("Cannot increase the value of V_low pass 255")
                V_low = 255
            else:
                V_low +=1
                print(f"Currently increasing V_low: Value = {V_low}")
        elif cur_mode == 4:
            if H_high >=179:
                print("Cannot increase the value of H_high pass 179")
                H_high = 179
            else:
                H_high += 1
                print(f"Currently increasing H_high: Value = {H_high}")
        elif cur_mode == 5:
            if S_high >=255:
                print("Cannot increase the value of S_high pass 255")
                S_high = 255
            else:
                S_high += 1
                print(f"Currently increasing S_high: Value = {S_high}")
        elif cur_mode == 6:
            if V_high >= 255:
                print("Cannot increase the value of V_high pass 255")
                V_high = 255
            else:
                V_high += 1
                print(f"Currently increasing V_high: Value = {V_high}")
        else:
            print("Please select the mode with button B first! Number 2 on keyboard!")

    # When Y button is pressed, decrease SPEED or ANGLE divisor by 1 (0 < d < 10)
    if rc.controller.is_down(rc.controller.Button.Y):
        if cur_mode == 1:
            if H_low <= 0:
                print("Cannot decrease the value of H_low pass 0")
                H_low = 0
            else:
                H_low -= 1
                print(f"Currently decreasing H_low: Value = {H_low}")
        elif cur_mode == 2:
            if S_low <= 0:
                print("Cannot decrease the value of S_low pass 0")
                S_low = 0
            else:
                S_low -= 1
                print(f"Currently decreasing S_low: Value = {S_low}" )
        elif cur_mode == 3:
            if V_low <= 0:
                print("Cannot decrease the value of V_low pass 0")
                V_low = 0
            else:
                V_low -=1
                print(f"Currently decreasing V_low: Value = {V_low}")
        elif cur_mode == 4:
            if H_high <= 0:
                print("Cannot decrease the value of H_high pass 0")
                H_high = 0
            else:
                H_high -= 1
                print(f"Currently decreasing H_high: Value = {H_high}")
        elif cur_mode == 5:
            if S_high <= 0:
                print("Cannot decrease the value of S_high pass 0")
                S_high = 0
            else:
                S_high -= 1
                print(f"Currently decreasing S_high: Value = {S_high}")
        elif cur_mode == 6:
            if V_high <= 0:
                print("Cannot decrease the value of V_high pass 0")
                V_high = 0
            else:
                V_high -= 1
                print(f"Currently decreasing V_high: Value = {V_high}")
        else:
            print("Please select the mode with button B first! Number 2 on keyboard!")

    # When left bumper is pressed, print HSV Values to screen
    if rc.controller.was_pressed(rc.controller.Button.LB):
        print(f"Current HSV Values: ({H_low}, {S_low}, {V_low}), ({H_high}, {S_high}, {V_high})")

    # When right bumpter is pressed, reset all HSV Values
    if rc.controller.was_pressed(rc.controller.Button.RB):
        # Reset the values
        H_low, H_high = 0, 179
        S_low, S_high = 0, 255
        V_low, V_high = 0, 255
        
        print("HSV values reset!")


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
