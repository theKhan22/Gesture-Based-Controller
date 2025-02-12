# Gesture-Based Control System

This project is a gesture-based control system that utilizes computer vision and machine learning to recognize hand gestures and translate them into corresponding actions.

## Installation

To set up and run this project, install the required dependencies:

```sh
pip install numpy opencv-python mediapipe pyautogui google.protobuf pynput comtypes
```

## Required Libraries

Ensure you have the following libraries installed:

```python
import numpy as np
import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import subprocess
from time import sleep
from google.protobuf.json_format import MessageToDict
from pynput.keyboard import Controller, Key
```

## Usage

Run the program using:

```sh
python start.py
```

## Features

- Real-time gesture detection using MediaPipe
- Control computer functions using hand gestures
- Uses OpenCV for image processing

## License

This project is open-source and available for modification and distribution.
