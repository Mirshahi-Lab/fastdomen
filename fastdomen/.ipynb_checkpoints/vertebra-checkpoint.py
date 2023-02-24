import cupy as cp
from cupyx.scipy import ndimage
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import torch

from fastdomen.imaging.utils import normalize_pytorch, normalize_zero_one


def detect_vertebra_slices(ds, model, vert_weights, output_dir):
    locations = {}
    frontal = ds.get_mip(axis=1)
    original_y = frontal.shape[0]
    torch_frontal = preprocess_for_detection(frontal, ds.spacing, 1)
    with torch.no_grad():
        for vert, weight in vert_weights.items():
            model.load_state_dict(torch.load(weight), strict=False)
            pred = model(torch_frontal)
            loc, prob, confidence = postprocess(pred, original_y)
            locations[vert] = {'slice_idx': loc, 'probability': prob, 'confidence': confidence}
    plot_vertebra_overlay(frontal, locations, output_dir, ds.filename)
    torch.cuda.empty_cache()
    return locations


def reduce_hu_intensity_range(img, minv=100, maxv=1500):
    img = cp.clip(img, minv, maxv)
    img = 255 * normalize_zero_one(img)
    return img


def preprocess_for_detection(image, spacing, target_spacing):
    image = ndimage.zoom(image.copy(), [spacing[2] / target_spacing, spacing[0] / target_spacing])
    image = reduce_hu_intensity_range(image)
    image = cp.asarray(resize(image.get(), (512, 512), preserve_range=True))
    image = normalize_pytorch(image, image.max())[cp.newaxis, cp.newaxis, ...]
    image = torch.as_tensor(image, dtype=torch.float32).cuda()
    return image


def postprocess(pred, orig_dim):
    pred = cp.asarray(torch.squeeze(pred))
    pred = pred.round(5)
    prob = round(float(pred.max()), 2)
    y_loc = cp.where(pred == pred.max())[0]
    if len(y_loc) > 1:
        y_loc = y_loc[0]
    y_loc = int(cp.floor(y_loc / 512 * orig_dim))
    if prob >= 0.5:
        confidence = 'high'
    elif prob < 0.5 and prob > 0.2:
        confidence = 'medium'
    else:
        confidence = 'low'
    return y_loc, prob, confidence
    

def plot_vertebra_overlay(frontal, locations, output_dir, filename):
    color_dict = {
        'high': 'g',
        'medium': 'y',
        'low': 'r'
    }
    outfile = f'{output_dir}/{filename}_vertebra_detection.jpg'
    plt.imshow(frontal.get())
    for vert, loc in locations.items():
        idx = loc['slice_idx']
        prob = loc['probability']
        confidence = loc['confidence']
        color = color_dict[confidence]
        plt.axhline(idx, color=color, lw=1)
        plt.text(x=20, y=idx-5, s=f'{vert}: prob={prob}, idx={idx}', color='w')
    plt.savefig(outfile, dpi=150)
    plt.close()