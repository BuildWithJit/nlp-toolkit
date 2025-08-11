from __future__ import annotations
from PIL import Image
import numpy as np
import cv2

def preprocess_pil(image:Image.Image, *, deskew:bool= True, denoise:bool= True) -> Image.Image:
    a = np.array(image.convert("L"))
    if deskew:
        a = _deskew(a)
    if denoise:
        a = cv2.fastNlMeansDenoising(a, h=10)
    return Image.fromarray(a)

def _deskew(a:np.ndarray) -> np.ndarray:
    thresold = cv2.threshold(a,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresold > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h,w = a.shape
    M =  cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv2.warpAffine(a, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)