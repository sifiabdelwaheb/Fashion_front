import dash
import datetime
import logging
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
import base64
from PIL import Image
from dash.exceptions import PreventUpdate
import urllib
import cv2
from urllib.request import urlopen
import io
from scipy import misc
import imageio
import numpy as np
import tensorflow as tf
logger = logging.getLogger(__name__)


new_model = tf.keras.models.load_model('fashionmodel.h5')
import cv2
import base64
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
ii = cv2.imread("nG8Kj.jpg")
lower_blue = np.array([0,0,0])
upper_blue = np.array([112,122,120])
gray_images = cv2.cvtColor(ii, cv2.COLOR_BGR2HSV)

gray_image = cv2.inRange(gray_images, lower_blue, upper_blue)

print(gray_image.shape)
plt.imshow(gray_image)
width = int(28)
height = int(28)
dim = (width, height)
# resize image
resized = cv2.resize(gray_image, dim, interpolation = cv2.INTER_AREA)
resized = resized.astype('float32') / 255
image = resized.reshape(1, 28, 28, 1)
import numpy as np
predictions = new_model.predict(image)

# Print our model's predictions.
pred=np.argmax(predictions, axis=1)
print(pred)
# Check our predictions against the !ground truths.
