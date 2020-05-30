import base64
import io

import cv2
import numpy as np
from PIL import Image

def b64toOCVImg(base64_img):
    picture_raw = base64.b64decode(base64_img)
    im_arr = np.frombuffer(picture_raw, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def b64toPILImg(base64_img):
    picture_raw = base64.b64decode(base64_img)
    image = Image.open(io.BytesIO(picture_raw))
    return image