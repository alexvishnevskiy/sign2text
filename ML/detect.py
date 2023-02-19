from cvzone.HandTrackingModule import HandDetector
import cv2
import torch
from torchvision.transforms import (
    Compose, RandomHorizontalFlip, RandomVerticalFlip,
    Resize, Normalize, ToTensor, PILToTensor,
    RandomRotation, RandomAutocontrast, RandomRotation
    )
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def convert2letter(label):
    mapping = {0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    26: 'del',
    27: 'nothing',
    28: 'space'}
    return mapping[label]

def get_model(path):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 29)
    model.load_state_dict(torch.load(path))
    model.cuda()
    model.eval()
    return model

def get_aug():
    valid_aug = Compose([
        Resize((128, 128)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    return valid_aug

def detect(img_path, model_path):
    img = cv2.imread(img_path)
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    hands = detector.findHands(img, draw=False)
    aug = get_aug()
    model = get_model(model_path)
    offset = 100

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        im = Image.fromarray(imgCrop)

        with torch.no_grad():
            out = model(aug(im).unsqueeze(0).cuda())
        label = torch.argmax(out).cpu().item()
        return convert2letter(label)
    else:
        return ""
