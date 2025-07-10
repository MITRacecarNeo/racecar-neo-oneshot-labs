# racecar-neo-oneshot-labs
Utility and quick-demo scripts for Autonomous RACECAR Neo

## Demo Files
For basic test and verification files located in **labs**, see below:
- **demo.py**: When run, press the "A" button to show text to terminal, and the "B" button to drive forward and to the right for 1 second.
- **test_core.py**: This file should be run with a monitor (no headless), when the "A", "X", and "Y" buttons are pressed, the camera, LIDAR, and IMU data are shown respectively.
- **test_async.ipynb**: This file should be run with a computer (headless), tests to make sure that the ROS-Jupyter async data pipeline is working properly. You should be able to run the cells and see images of the data from sensors pop up.

For more advanced demo files for workshops and tests, files are located in **labs/utility**, see below for explanations per files: 
- **hsv_tuner.py**: Provides a UI for tuning HSV values for color image segmentation. Can be used in sim (no mac) or on the car (with monitor).
- **hsv_tuner_non_gui.py**: For mac devices, multiple threaded UI windows are not supported. The HSV tuning interface is provided through the terminal instead.
- **hsv-p_tuner.py**: Same as hsv-p_tuner, but two additional trackbars allow user to tune the car's autonomy with a basic P-type controller.
- **ss-pd_tuner.py**: Safety Stop Precision Driving tuner, allows for trimming of the car's angle and tuning of a LIDAR-based safety stop controller. Can be used in the sim (no mac) or on the car (with monitor).
- **lfss.py**: Line Following with Safety Stop tuner, assumes user is proficient with tuning the **hsv-p_tuner.py** and **ss-pd_tuner.py** files. Insert parameters to run on the vehicle and perform basic sensor fusion to follow a line and stop when an obstacle is detected.
- **steering_trim**: Basic steering calibration for the vehicle. ***Caution***: Overwrites current pwm.py values and kills teleop!!
- **lagmachine.py**: (Advanced) Implements an artificial delay between frames for line following to practice tuning a delay compensation controller.