"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: ss-pd_tuner.py

Title: Safety Stop Precision Driving Tuner

Author: Chris Lai (MITLL)

Purpose: Provide the end-user with a UI that has the following capabiitlies:
- Able to manually drive the vehicle forwards, backwards, left, and right
- Able to change the speed of the vehicle using trackbars
- Able to change the angle offset of the vehicle in precision to the hundreths place
- Enables autonomous safety stop features, which include:
    - Proportional controller for speed reduction dependent on distance from object
    - Setpoint and threshold settings for the controller
    - LIDAR forward detection angle/window to which average is taken of

UI contains the following editable trackbars:
- Speed Setting: 0 to 100 (maps 0 to 1)
- Angle Adjustment: -100 to 100 (maps to 0.0025 increment, -0.25 to 0.25 offset)
- Setpoint Adjustment: 0cm to 100cm
- Sensitivity Adjustment: 0% to 100%
- LIDAR Angle Adjustment: 0deg to 45deg (mirrored on each side)
"""

import sys
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import RACECAR Library from local file path
sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

# Create RACECAR object
rc = racecar_core.create_racecar()

# Default Variables
global speed, angle
global tune_speed, angle_offset
tune_speed = 0.8 # out of 1
angle_offset = 0

global setpoint, kp, lidar_angle
setpoint = 50 # cm
kp = 0
lidar_angle = 30 # total angle (both sides)

# Variables for data logging and mapping
global loc_history
hist_len = 250 # save this many frames
loc_history = [0] * hist_len

# [FUNCTION] UI -> RACECAR variable mapping
def on_speed_change(val):
    global tune_speed
    tune_speed = float(val)/100

def on_angle_offset_change(val):
    global angle_offset
    angle_offset = float(val) * 0.0025

def on_setpoint_change(val):
    global setpoint
    setpoint = float(val) # 1:1 conversion in cm

def on_kp_change(val):
    global kp
    kp = float(val) # 1:1 conversion as %percentage

def on_lidar_angle_change(val):
    global lidar_angle
    lidar_angle = float(val)

# [FUCNTION] Create a GUI with trackbars for the indicated parameters
def create_gui():
    root = tk.Tk()
    root.title("Safety Stop Precision Driving Tuner")
    root.geometry("400x500")
    root.configure(background="black")

    # Create custom fonts
    title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
    subtitle_font = tkfont.Font(family="Helvetica", size=12, weight="bold", slant="italic")

    # Create and pack the title label
    title_label = tk.Label(root, text="RACECAR Neo Demo UI", background='black', foreground='red', font=title_font)
    title_label.pack()

    # Create and pack the subtitle label
    subtitle_label = tk.Label(
        root,
        text="Safety Stop Precision Driving Tuner App",
        background='black',
        foreground='orange',
        font=subtitle_font
    )
    subtitle_label.pack()

    # Configure the ttk style for a dark theme
    style = ttk.Style()
    style.theme_use('clam')

    # Define a custom font
    custom_font = tkfont.Font(family="Helvetica", size=10, weight="bold")
    
    # Configure the background and foreground of the Scale widget and labels
    style.configure(
        "Horizontal.TScale",
        background='black',
        troughcolor='grey',
        bordercolor='black',
        lightcolor='black',
        darkcolor='black',
        arrowcolor='white'
    )
    style.configure("TLabel", background='black', foreground='white', font=custom_font)
    style.configure("Outline.TFrame", background='black', borderwidth=2, relief='solid')

    # Attempt to change the thumb using element create
    style.element_create('custom.Horizontal.Scale.slider', 'from', 'default')
    style.layout('Custom.Horizontal.TScale', [
        ('Horizontal.Scale.trough', {
            'sticky': 'nswe',
            'children': [('custom.Horizontal.Scale.slider', {'side': 'left', 'sticky': ''})]
        })
    ])
    style.configure(
        'Custom.Horizontal.TScale',
        sliderrelief='flat',
        sliderlength=30,
        slidershadow='black',
        slidercolor='blue'
    )

    # Adjust styles to include black borders and red trough
    style.configure("Horizontal.TScale", background='black', troughcolor='grey', bordercolor='white')

    # Function to update label
    def update_label(label, value):
        label.config(text=f"{int(float(value))}")

    # Creating a scale widget with a frame
    def create_scale(root, label, from_, to, start, command):
        frame = ttk.Frame(root, style="Outline.TFrame")
        frame.pack(fill='x', expand=True, pady=10)

        min_label = tk.Label(frame, text=str(from_), bg='black', fg='white', font=custom_font)
        min_label.pack(side='left')

        max_label = tk.Label(frame, text=str(to), bg='black', fg='white', font=custom_font)
        max_label.pack(side='right')

        current_value_label = tk.Label(frame, text=str(from_), bg='black', fg='white', font=custom_font)
        current_value_label.pack(side='top')

        scale = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            orient='horizontal',
            style="Horizontal.TScale",
            command=lambda v, l=current_value_label: (command(v), update_label(l, v))
        )
        scale.set(start)  # Default value
        scale.pack(fill='x', expand=True, padx=4, pady=4)

        var_name_label = tk.Label(frame, text=label, bg='black', fg='white', font=custom_font)
        var_name_label.pack(side='bottom')

        return scale
    
    # Create scales for trackbars
    create_scale(root, "Speed (%)", 1, 100, 80, on_speed_change)
    create_scale(root, "Angle Offset (x0.0025)", -100, 100, 0, on_angle_offset_change)
    create_scale(root, "Setpoint (cm)", 1, 200, 100, on_setpoint_change)
    create_scale(root, "Sensitivity (%)", 1, 100, 50, on_kp_change)
    create_scale(root, "LIDAR Window (deg)", 1, 90, 45, on_lidar_angle_change)

    root.mainloop()

# [FUCTION] Function to graph data to screen - threaded
def graph_data():
    fig, ax = plt.subplots()
    line, = ax.plot(range(hist_len), list(loc_history), label="Line Position")
    setpoint_line = ax.axhline(y=int(setpoint), color='red', linestyle='--', label='Setpoint')

    # Set title and label
    ax.set_title("Plot of Distance from Wall vs. Frame #")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Distance (cm)")

    def update_plot(frame):
        line.set_ydata(loc_history)
        setpoint_line.set_ydata(int(setpoint))
        return line, setpoint_line
    
    # Set plot limits
    ax.set_ylim(0, int(setpoint) * 3)
    ax.set_xlim(0, int(hist_len) - 1)

    # Add legend
    ax.legend()

    # Start animation
    ani = animation.FuncAnimation(fig, update_plot, interval=33, blit=True)

    # Show plot
    plt.show()

# [FUNCTION] Start function
def start():
    global speed, angle
    # Set initial driving speed and angle
    speed = 0
    angle = 0
    rc.drive.set_speed_angle(speed, angle)

    # Thread it
    gui_thread = threading.Thread(target=create_gui)
    gui_thread.start()
    graph_thread = threading.Thread(target=graph_data)
    graph_thread.start()

    # Print start message
    print(
        ">> RACECAR Neo OneShot Demo - Safety Stop with Precision Tuning\n"
        "\n"
        "Controls:\n"
        "   Right Bumper = RACECAR Dead Man's Switch\n"
        "   A button = print current speed and angle to the terminal window\n"
        "   B button = print out current system parameters"
    )

# [FUNCTION] Update function
def update():
    global speed 
    global angle

    # Manual Controller Paradigm (optional)
    a, b = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    x, y = rc.controller.get_joystick(rc.controller.Joystick.RIGHT)

    if b > 0.1 or b < -0.1:
        speed = rc_utils.clamp(speed, -tune_speed, tune_speed)
    else:
        speed = 0

    if x > 0.1 or x < -0.1:
        angle = x
    else:
        angle = 0

    # LIDAR data retrieval & controller
    scan = rc.lidar.get_samples()
    window = (360-lidar_angle/2, lidar_angle/2)
    loc_angle, distance = rc_utils.get_lidar_closest_point(scan, window)
    error = setpoint - distance
    kp_now = -1/setpoint * kp/100 * 2

    # Automatically adjust speed based on error if error > setpoint * 2
    if error > -setpoint:
        speed = kp_now * error
        speed = rc_utils.clamp(speed, -tune_speed, tune_speed)
        if b < -0.1:
            speed = rc_utils.clamp(b, -tune_speed, tune_speed)
    else:
        speed = rc_utils.clamp(b, -tune_speed, tune_speed)

    # Angle offset modifier
    angle += angle_offset
    angle = rc_utils.clamp(angle, -1, 1)

    # Location history graph update
    if len(loc_history) >= hist_len: 
        loc_history.pop(-1)
    loc_history.insert(0, distance)

    # Print debug statement
    print(f"Distance to wall: {round(distance,2)} || Kp: {kp_now} || Speed: {round(speed,2)} || Angle: {round(angle,2)} || Error: {round(error,2)}")
    
    # Send speed and angle to the car if trigger is pressed
    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0:
        rc.drive.set_speed_angle(speed, angle)
    else:
        rc.drive.set_speed_angle(0, 0)

    # Display LIDAR to screen
    rc.display.show_lidar(scan, max_range=500)

    ######################
    # CONTROLLER OPTIONS #
    ######################

    # Print statements from buttons
    if rc.controller.is_down(rc.controller.Button.A):
        print(f"Speed: {speed} || Angle: {angle}")
    
    if rc.controller.was_pressed(rc.controller.Button.B):
        print(f"Speed coefficient: {tune_speed}% || Angle offset: {angle_offset*0.0025} || Setpoint: {setpoint}cm || Sensitivity: {kp}% || LIDAR Window: {lidar_angle}")


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
