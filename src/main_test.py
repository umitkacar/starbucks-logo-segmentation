import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import json
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate

from mobile_seg.dataset import load_df, MaskDataset
from mobile_seg.modules.net import load_trained_model

from pathlib import Path

import imgviz
import imageio

def visualize(compared_file, images_save=True, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap=plt.cm.gray)

    if images_save:
        plt.savefig(compared_file)

if __name__ == '__main__':

     # Output
    OUTPUT_DIR = Path('../output')
    OUTPUT_DIR.mkdir(exist_ok=True)
    VISUAL_DIR = OUTPUT_DIR / "visual"
    VISUAL_DIR.mkdir(exist_ok=True)
    GIF_DIR = OUTPUT_DIR / "gif_image"
    GIF_DIR.mkdir(exist_ok=True)

    with open(f'params/config.json') as f:
        config = json.load(f)

    model = load_trained_model(config).to(config["device"]).eval()
    
    with torch.no_grad():

        img_path = Path("../../datahub/starbucks/image/bagir-bahana-6295IjcQkSQ-unsplash.jpg")

        inputs = cv2.imread(str(img_path))
        #inputs = cv2.imread("../../datahub/starbucks/test_wild/test-3.jpeg")
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        inputs = cv2.resize(inputs, (config["img_size"],config["img_size"]),interpolation = cv2.INTER_NEAREST)
        inputs = np.array(inputs).astype(np.float32).transpose((2, 0, 1))
        inputs = inputs/255.
        inputs = torch.from_numpy(inputs)
        inputs = inputs.unsqueeze(0)

        outputs = model(inputs.to(config["device"])).cpu()
        outputs = outputs.squeeze()
        
        inputs = inputs.squeeze()
        inputs = (inputs * 255).numpy().transpose((1, 2, 0)).astype(np.uint8)
    
        path = VISUAL_DIR / f"Starbucks_logo_{img_path.stem}.png"
        img_save = True
        visualize ( path, img_save,
                    img_orj       = inputs,
                    mask_predict  = outputs)

        # colorize label image
        class_label = outputs.squeeze().numpy().astype(int)
        labelviz = imgviz.label2rgb(class_label,
                            image=inputs, 
                            #label_names=['background','starbucks_logo'],
                            alpha = 0.5,
                            font_size=30,
                            colormap=imgviz.label_colormap(n_label=256, value=255),
                            loc="rb",)

        #plt.figure(figsize=(80, 5))
        count = 1
        plt.imshow(labelviz)
        image_name_orj = GIF_DIR / "gif_frame_1.jpg"
        plt.imsave(image_name_orj, inputs)
        count = count +1
        image_name = GIF_DIR / "gif_frame_2.jpg"
        plt.imsave(image_name,labelviz)

        imageio.plugins.freeimage.download()

        anim_file = OUTPUT_DIR / f'Starbucks_logo_{img_path.stem}.gif'

        filenames = GIF_DIR.glob("gif_frame_*.jpg")
        filenames = sorted(filenames)
        last = -1
        images = []
        for filename in filenames:
            image = imageio.imread(filename)
            images.append(image)

        imageio.mimsave(anim_file, images, 'GIF-FI', fps=2)
