from PIL import Image
import numpy as np

def flatten_image(filename):
    im = Image.open(filename)
    list(im.getdata())  # Since im was in rgba, this will output a
    # sequence of quintuples [(0, 0, 0, 0), (0, 0, 0, 0), ...
    im_array = np.fromstring(im.tobytes(), dtype=np.uint8)# 1-dimensional
    return im_array