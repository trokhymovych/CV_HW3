import numpy as np


def ssd(im1, im2):
    return np.sum((im1 - im2) ** 2)


def sad(im1, im2):
    return np.sum(np.abs(im1 - im2))


def normalize(im):
    im0 = im - np.mean(im)
    return (im0) / np.linalg.norm(im0)


def ncc(im1, im2):
    return np.sum(normalize(im1) * normalize(im2))


def match_template(image, template, method):
    h_temp, w_temp, _ = template.shape
    h_im, w_im, _ = image.shape
    res = np.zeros(shape=(h_im - h_temp + 1, w_im - w_temp + 1))
    for x in range(res.shape[1]):
        for y in range(res.shape[0]):
            img_part = image[y:y + h_temp, x:x + w_temp]
            res[y, x] = method(img_part, template)
    return res
