"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: control.py

Title: Control Testing

Author: Chris Lai (MITLL)

Purpose: Provide users with the ability to rapidly switch between controllers to observe
the different behaviors of the autonomous car as it follows a line.

Expected Outcome: When the buttons are pressed, change the type of controller that is
turning the RACECAR for a line follower problem:
- When the A button is pressed, no controller is active
- When the B button is pressed, set the controller to bang-bang control
- When the X button is pressed, set the controller the proportional control
- When the Y button is pressed, set the controller to PD control
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

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 30

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# HSV pair for blue (Replace this with your tuned values!)
BLUE = ((90, 80, 120), (120, 255, 255))  # The HSV range for the color blue

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
prev_error = 0  # The previous error saved

# Drive Mode Enumeration:
# 0: None
# 1: Bang-Bang
# 2: Proportional Control
# 3: PD Control
drive_mode = 0


########################################################################################
# Functions
########################################################################################

# [FUNCTION] Finds contours in the current color image and uses them to update 
# contour_center and contour_area
def update_contour():
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

        # Find all of the contours of the current color
        contours = rc_utils.find_contours(image, BLUE[0], BLUE[1])

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)

        # Display the image to the screen
        rc.display.show_color_image(image)


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():
    global speed
    global angle
    global drive_mode
    global prev_error

    # Set the driving mode based on the buttons pressed (0 default)
    if rc.controller.was_pressed(rc.controller.Button.A):
        drive_mode = 0
        print(f"Set Drive Mode to 0: No controller")
    if rc.controller.was_pressed(rc.controller.Button.b):
        drive_mode = 1
        print(f"Set Drive Mode to 1: Bang-bang controller")
    if rc.controller.was_pressed(rc.controller.Button.X):
        drive_mode = 2
        print(f"Set Drive Mode to 2: Proportional controller")
    if rc.controller.was_pressed(rc.controller.Button.Y):
        drive_mode = 3
        print(f"Set Drive Mode to 3: PD controller")

    # Define current frame constants for calculating closed-loop controller
    setpoint = 320
    error = setpoint - contour_center[1]
    de_dt = (error - prev_error) / rc.get_delta_time()
    prev_error = error

    # Define PD coefficients
    KP = 1 / 320  # normalize screen pixels to angle range
    KD = 0.01  # default value, will need tuning later

    # Define behaviors for calculating angle based on each drive mode
    if drive_mode == 0:  # Nothing happens
        angle = 0
    elif drive_mode == 1:  # Bang-bang control
        if error < 0:
            angle = 1
        elif error > 0:
            angle = -1
        else:
            angle = 0
    elif drive_mode == 2:  # Proportional control
        angle = KP * error
        # Saturate angle in case goes over bounds
        if angle > 1.0:
            angle = 1.0
        elif angle < -1.0:
            angle = -1.0
    elif drive_mode == 3:  # PD control
        angle = KP * error + KD * de_dt
        # Saturate angle in case goes over bounds
        if angle > 1.0:
            angle = 1.0
        elif angle < -1.0:
            angle = -1.0

    # Set a comfortable speed for driving
    speed = 1 / 8.5  # May need to be tuned!!

    # Send speed and angle to the car
    print("Speed:", speed, "Angle:", angle)
    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
