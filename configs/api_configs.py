import numpy as np
import io
import cv2
# from PIL import Image
import torch

# ================================================================= #

def  get_image_from_url(image_content: bytes) -> np.ndarray:
    image_stream = io.BytesIO(image_content)
    image = cv2.imdecode(np.frombuffer(image_stream.getvalue(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image