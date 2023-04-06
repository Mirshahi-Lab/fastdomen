import os
import torch
import torch.nn as nn
import torchvision.models as models
import cupy as cp
import numpy as np
from skimage.transform import resize
from fastdomen.imaging.dicomseries import DicomSeries
from fastdomen.imaging.utils import normalize_pytorch, normalize_zero_one


def preprocess(image):
    im = resize(image.get(), (224, 224))
    im = 255 * normalize_zero_one(im)
    im = cp.asarray(normalize_pytorch(im, im.max(), 0.445, 0.269))
    inp = cp.stack([im, im, im])
    inp = cp.expand_dims(inp, 0)
    inp = torch.as_tensor(inp, device='cuda')
    return inp


def predict(im, model):
    with torch.no_grad():
        pred = model(im)
        pred = torch.sigmoid(pred)
    return round(pred.item(), 4)


def recog_ct(ds: DicomSeries):
     # Load the model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    # model.fc = model.fc.cuda()
    model = model.cuda()
    model.eval()
    im = ds.frontal.copy()
    im = preprocess(im)
    model.load_state_dict(torch.load('fastdomen/recog_models/body_v_non_body.pth'))
    body_non_body = predict(im.clone(), model)
    if body_non_body > 0.5:
        return 'other'
    
    model.load_state_dict(torch.load('fastdomen/recog_models/chest_v_abd.pth'))
    chest_v_abd = predict(im.clone(), model)
    if chest_v_abd < 0.5:
        return 'abdomen'
    
    model.load_state_dict(torch.load('fastdomen/recog_models/chest_v_body.pth'))
    chest_v_body = predict(im.clone(), model)
    if chest_v_body > 0.5:
        return 'abdomen'
    else:
        return 'chest'
