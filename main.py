import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pygame as pg

from utils import test, show_results, var_blur, launch_interface

def main(save_model,load_model):
    # #READING IMAGE FROM PATH
    # img_path = "./sample3.jpg"
    # img = cv.imread(img_path)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    launch_interface()


if __name__ == "__main__":
    save_model = False
    load_model = True
    main(save_model, load_model)