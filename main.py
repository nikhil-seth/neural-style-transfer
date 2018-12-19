"""
Importing Libraries : 
    Numpy -> Numerical Computation
    matplotlib -> Plotting
    tensorflow -> VGG19 model
    tensorflow.contrib.earger -> Eager Execution
    PIL -> Help in Image Importing
"""
import numpy as np
import matplotlib as mpl
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from funct import *
import IPython.display

# Enabling Eager Execution
tf.enable_eager_execution()
""" Setting paths for Images"""
content_path = 'images/Green_Sea_Turtle_grazing_seagrass.jpg'
style_path = 'images/The_Great_Wave_off_Kanagawa.jpg'

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
best, best_loss = run_style_transfer(content_path, 
                                     style_path, content_layers, style_layers,
                                     num_iterations=1000)


image_to_array(best)
show_results(best, 'images/Green_Sea_Turtle_grazing_seagrass.jpg',
             'images/The_Great_Wave_off_Kanagawa.jpg')

