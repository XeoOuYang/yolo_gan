import os

import cv2
from PIL import Image
from PIL.Image import Transpose


def get_temp(path, tmp='_tmp'):
    dir_name = os.path.dirname(path)
    name, suffix = os.path.splitext(os.path.basename(path))
    return os.path.join(dir_name, f'{name}{tmp}{suffix}')

def chg_suffix(path, suffix, base_dir=''):
    dir_name = base_dir if base_dir != '' else os.path.dirname(path)
    name, _ = os.path.splitext(os.path.basename(path))
    return os.path.join(dir_name, f'{name}{suffix}')

def mosaic_magic(source, ratio: float=0.1):
    image = Image.open(source)
    w = image.width
    h = image.height
    w_p = int(w*ratio)
    h_p = int(h*ratio)
    dst_image = Image.new(image.mode, (w+2*w_p, h+2*h_p))
    # 中间
    dst_image.paste(image, (w_p, h_p))
    # 水平翻转
    h_image = image.transpose(Transpose.FLIP_LEFT_RIGHT)
    dst_image.paste(h_image.crop((0, 0, w_p, h)), (w+w_p, h_p))
    dst_image.paste(h_image.crop((w-w_p, 0, w, h)), (0, h_p))
    # 垂直翻转
    v_image = image.transpose(Transpose.FLIP_TOP_BOTTOM)
    dst_image.paste(v_image.crop((0, 0, w, h_p)), (w_p, h+h_p))
    dst_image.paste(v_image.crop((0, h-h_p, w, h)), (w_p, 0))
    # 中心翻转
    c_image = image.transpose(Transpose.FLIP_LEFT_RIGHT).transpose(Transpose.FLIP_TOP_BOTTOM)
    dst_image.paste(c_image.crop((0, 0, w_p, h_p)), (w+w_p, h+h_p))
    dst_image.paste(c_image.crop((w-w_p, h-h_p, w, h)), (0, 0))
    dst_image.paste(c_image.crop((w-w_p, 0, w, h_p)), (0, h+h_p))
    dst_image.paste(c_image.crop((0, h-h_p, w_p, h)), (w+w_p, 0))
    # 保存文件
    tmp_path = get_temp(source)
    dst_image.save(tmp_path)
    return tmp_path

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # Normalize pixel values
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # return img
    return img

def check_cuda_env():
    import ultralytics
    ultralytics.checks()
