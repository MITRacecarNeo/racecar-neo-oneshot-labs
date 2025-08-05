"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-oneshot-labs

File Name: wall-follow_tuner.py

Title: Wall Following Tuner

Author: Chris Lai (MITLL)

Purpose: Provide the end-user with a UI that has the following capabiitlies:
- Able to autonomously drive a wall-follower when the right bumper and trigger are pressed.
- Able to change the speed of the vehicle using trackbars
- Implements a basic P-type controller that reduces cross-track error between two walls
- Able to change the Kp sensitivity gain of the controller
- Able to change the LIDAR closest-point window from +/- 90 degree ranges

UI contains the following editable trackbars:
- Speed Setting: 0 to 100 (maps 0 to 1)
- Sensitivity Adjustment: 0% to 100% (maps to Kp values of -0.02 to 0.02)
- LIDAR Angle Adjustment: 0deg to 90deg (around +/- 90 degree range)
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
global tune_speed
tune_speed = 0.8 # out of 1
angle_offset = 0

global kp, lidar_angle
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

def on_kp_change(val):
    global kp
    kp = float(val) # 1:1 conversion as %percentage

def on_lidar_angle_change(val):
    global lidar_angle
    lidar_angle = float(val)

# [FUCNTION] Create a GUI with trackbars for the indicated parameters
def create_gui():
    root = tk.Tk()
    root.title("Wall Following Tuner")
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
        text="Wall Following Tuner App",
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
    create_scale(root, "Sensitivity (%)", 1, 100, 50, on_kp_change)
    create_scale(root, "LIDAR Window (deg)", 1, 90, 45, on_lidar_angle_change)

    root.mainloop()

# [FUCTION] Function to graph data to screen - threaded
def graph_data():
    fig, ax = plt.subplots()
    line, = ax.plot(range(hist_len), list(loc_history), label="Line Position")
    setpoint_line = ax.axhline(y=int(0), color='red', linestyle='--', label='Setpoint')

    # Set title and label
    ax.set_title("Plot of Distance from Wall vs. Frame #")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Distance (cm)")

    def update_plot(frame):
        line.set_ydata(loc_history)
        setpoint_line.set_ydata(int(0))
        return line, setpoint_line
    
    # Set plot limits
    ax.set_ylim(-100, 100)
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
        ">> RACECAR Neo OneShot Demo - Wall Following Tuner\n"
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

    # LIDAR data retrieval & controller
    scan = rc.lidar.get_samples()
    left_window = (270-lidar_angle, 270+lidar_angle)
    right_window = (90-lidar_angle, 90+lidar_angle)
    left_loc_angle, left_distance = rc_utils.get_lidar_closest_point(scan, left_window)
    right_loc_angle, right_distance = rc_utils.get_lidar_closest_point(scan, right_window)
    error = right_distance - left_distance
    kp_now = kp/10000 * 2
    angle = kp_now * error
    angle = rc_utils.clamp(angle, -1, 1)
    speed = tune_speed

    # Location history graph update
    if len(loc_history) >= hist_len: 
        loc_history.pop(-1)
    loc_history.insert(0, error)

    # Print debug statement
    print(f"Left Distance, Angle: {round(left_distance, 2)},{left_loc_angle} || Right Distance, Angle: {round(right_distance, 2)}, {right_loc_angle} || Error: {round(error, 2)} || Angle: {round(angle, 2)}")

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

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()