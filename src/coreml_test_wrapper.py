import coremltools as ct
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

Height = 512  # use the correct input image height
Width = 512
def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, Image.ANTIALIAS)
    return img

# load the image and resize using PIL utilities
img_path = "../abc.png"
img = load_image(img_path, resize_to=(Width, Height))
img.save('../results/imgtest.png')

mlmodel = ct.models.MLModel('../coreMLhub/mobileUNET/abc.mlmodel')
#mlmodel = ct.models.MLModel('../coreMLhub/mobileUNET_old/fingernailq16.mlmodel')

out_dict = mlmodel.predict({'input0': img})
out_dict["output"].save('../results/testOutBinary.png')

out = np.array(out_dict['output'].convert('L')).astype(np.uint8) # binaryß
# out = np.array(out_dict['output'].squeeze()).astype(np.uint8)
# out = np.array(np.argmax(out_dict['var_1406'].squeeze(),0)).astype(np.uint8)

PIL_image = Image.fromarray(out)
PIL_image.save('../results/testOut1.png')