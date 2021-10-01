import numpy as np
import matplotlib.pyplot as plt
import cv2
from white_balance import *
from rghs import *

def enhance(path):
    # path = path to image.
    org_img = cv2.imread(path)
    org_img = cv2.cvtColor(org_img,cv2.COLOR_BGR2RGB)
    wb = white_balance_1(org_img)
    out = rghs(wb)
    return out
