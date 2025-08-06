# General-purpose imports
import cv2
import numpy as np
import os
import sys
import time

# Racecar-specific imports
sys.path.insert(0, '../library')
import racecar_core
import racecar_utils as rc_utils

# PyCoral imports for object detection
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# Global variables
rc = racecar_core.create_racecar()

# Object detection variables
default_path = os.path.expanduser('~/jupyter_ws/TPS/labs/model')
model_name = 'machineVision.tflite'
label_name = 'labels.txt'
model_path = os.path.join(default_path, model_name)
label_path = os.path.join(default_path, label_name)
interpreter = None
labels = None
inference_size = None
SCORE_THRESH = 0.5
NUM_CLASSES = 1

# Controller variables
SPEED = 0.7  # Constant speed for the car

kp = 0.08  # Proportional gain
ki = 0.0  # Integral gain
kd = 0.1  # Derivative gain
prev_error = 0
integral = 0
last_time = 0

def start():
    """
    This function is run once every time the start button is pressed
    """
    global interpreter, labels, inference_size, last_time

    # Load the object detection model and labels
    print(f'Loading model: {model_path}')
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    labels = read_label_file(label_path)
    inference_size = input_size(interpreter)

    # Set the initial speed and angle
    rc.drive.set_speed_angle(0, 0)
    last_time = time.time()
    print(">> PID Controller Initialized")

def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Get the latest image from the camera
    image = rc.camera.get_color_image()

    if image is None:
        return

    # Preprocess the image for the model
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image_resized = cv2.resize(rgb_image, inference_size)

    # Run inference on the image
    run_inference(interpreter, rgb_image_resized.tobytes())
    objs = get_objects(interpreter, SCORE_THRESH)[:NUM_CLASSES]

    # Process the detected objects
    process_objects(image, objs)

    # Display the image
    rc.display.show_color_image(image)

def process_objects(image, objs):
    """
    Processes the detected objects to control the car.
    """
    global prev_error, integral, last_time

    if not objs:
        # If no objects are detected, stop the car
        rc.drive.stop()
        return

    # Assume the first detected object is the one to follow
    obj = objs[0]
    
    # Get the bounding box of the object
    height, width, _ = image.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    bbox = obj.bbox.scale(scale_x, scale_y)
    x0, y0 = int(bbox.xmin), int(bbox.ymin)
    x1, y1 = int(bbox.xmax), int(bbox.ymax)
    
    # Calculate the center of the bounding box
    center_x = (x0 + x1) // 2
    
    # Draw the bounding box and center dot
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.circle(image, (center_x, (y0 + y1) // 2), 5, (0, 0, 255), -1)

    # PID control logic
    image_center = rc.camera.get_width() // 2
    error = center_x - image_center

    # Calculate delta time
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    # Proportional term
    p_term = kp * error
    
    # Integral term
    integral += error * dt
    i_term = ki * integral

    # Derivative term
    derivative = (error - prev_error) / dt
    d_term = kd * derivative
    prev_error = error
    
    # Calculate the steering angle
    angle = (p_term + i_term + d_term) / 100
    angle = np.clip(angle, -1.0, 1.0) # Clamp the angle to the valid range

    # Set the car's speed and angle
    rc.drive.set_speed_angle(SPEED, angle)


if __name__ == '__main__':
    rc.set_start_update(start, update, None)
    rc.go()
