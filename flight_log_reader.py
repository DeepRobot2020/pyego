import argparse
import os
import numpy as np
import json
import errno
import pickle
import cv2
import math
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

MSG_NAMES = ['VISION_VO_MEAS', 'VISION_VO_DEBUG_INFO', 'AHRSSTATE', 'GPS_STATE', 'ACS_METADATA', 'SACS_']
NUM_LOG_HEADER = 3
