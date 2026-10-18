from PIL import Image
import numpy as np
from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer

def color_match(ref_img, src_img):
    cm = ColorMatcher() 
    img_ref_np = Normalizer(np.asarray(ref_img)).type_norm()
    img_src_np = Normalizer(np.asarray(src_img)).type_norm()

    img_res = cm.transfer(src=img_src_np, ref=img_ref_np, method='hm-mkl-hm')   # hm-mvgd-hm / hm-mkl-hm
    img_res = Normalizer(img_res).uint8_norm()
    img_res = Image.fromarray(img_res)
    return img_res