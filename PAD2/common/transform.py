import cv2
import math
import os
import random

import numpy as np

from PIL import Image


class SpatialBrightness(object):
    def __init__(self, brightness):
        self.brightness = brightness

    def generate_template(self, img_shape):
        template_h = np.ones(img_shape)
        template_w = np.ones(img_shape)
        for k in range(template_h.shape[1]):
            template_h[:, k] = k * 1.0 / template_h.shape[1]
        for k in range(template_w.shape[0]):
            template_w[k, :] = k * 1.0 / template_w.shape[0]

        return template_h, template_w

    def __call__(self, img):
        # if True:
        #     if not os.path.exists('before_aug.jpg'):
        #         img.save('before_aug.jpg')

        np_data = np.asarray(img).astype('float32')

        template_h, template_w = self.generate_template(np_data.shape)
        c = random.uniform(-self.brightness, self.brightness)
        rand_theta = random.randint(1, 359)

        # Rand Area Contrast
        h_rand = np.cos(rand_theta * 1.0 / 360 * 2.0 *  # Rand Select From[0, 360]
                        3.14159)
        w_rand = np.sin(rand_theta * 1.0 / 360 * 2.0 * 3.14159)

        if h_rand < 0:
            new_template_h = (1 - template_h) * h_rand * h_rand
        else:
            new_template_h = template_h * h_rand * h_rand

        if w_rand < 0:
            new_template_w = (1 - template_w) * w_rand * w_rand
        else:
            new_template_w = template_w * w_rand * w_rand

        np_data = np_data * (1 + (new_template_h + new_template_w) * c)
        np_data  = np.clip(np_data, 0, 255).astype('uint8')

        img = Image.fromarray(np_data)
        # if True:
        #     if not os.path.exists('after_aug.jpg'):
        #         img.save('after_aug.jpg')

        return img


class SpatialVariantBrightness(object):
    """Spatial variant brightness, Enhanced Edition.
    Powered by xin.wang@horizon.ai.

    Parameters
    ----------
    brightness : float, default is 0.6
        Brightness ratio for this augmentation, the value choice
        in Uniform ~ [-brightness, brigheness].
    max_template_type : int, default is 3
        Max number of template type in once process. Note,
        the selection process is repeated.
    """

    def __init__(self, brightness=0.6, max_template_type=3):
        self.brightness = brightness
        self.max_template_type = max_template_type

    def generate_template(self, h, w):
        # `sinwave` has a bigger proportion than others.
        temp_types = ['parabola', 'linear',
                      'qudratical', 'cubic', 'sinwave', 'sinwave']

        idxs = np.random.randint(
            0, len(temp_types), size=random.randint(1, self.max_template_type))

        temp_type_list = [temp_types[i] for i in idxs]
        template_h_list = []

        for temp_type in temp_type_list:
            template_h = np.ones((h, w))
            angle = random.randint(0, 360) * np.pi / 180.
            if temp_type == 'parabola':
                sin_x = math.sin(angle) ** 2 * (2 * (math.sin(angle) > 0) - 1)
                cos_x = math.cos(angle) ** 2 * (2 * (math.cos(angle) > 0) - 1)
                for x in range(w):
                    for y in range(h):
                        template_h[y, x] = ((sin_x*x/w + cos_x*y/h) - 0.5) ** 2
                min_value, max_value = np.min(template_h), np.max(template_h)
                template_h = (template_h - min_value) / (max_value - min_value)
            elif temp_type == 'linear':
                sin_x = math.sin(angle) ** 2 * (2 * (math.sin(angle) > 0) - 1)
                cos_x = math.cos(angle) ** 2 * (2 * (math.cos(angle) > 0) - 1)
                for x in range(w):
                    for y in range(h):
                        template_h[y, x] = (sin_x*x/w + cos_x*y/h)/2
                min_value, max_value = np.min(template_h), np.max(template_h)
                template_h = (template_h - min_value) / (max_value - min_value)
            elif temp_type == 'qudratical':
                sin_x = math.sin(angle) ** 2 * (2 * (math.sin(angle) > 0) - 1)
                cos_x = math.cos(angle) ** 2 * (2 * (math.cos(angle) > 0) - 1)
                for x in range(w):
                    for y in range(h):
                        template_h[y, x] = (sin_x*x/w + cos_x*y/h) ** 2
                min_value, max_value = np.min(template_h), np.max(template_h)
                template_h = (template_h - min_value) / (max_value - min_value)
            elif temp_type == 'cubic':
                sin_x = math.sin(angle) ** 2 * (2 * (math.sin(angle) > 0) - 1)
                cos_x = math.cos(angle) ** 2 * (2 * (math.cos(angle) > 0) - 1)
                for x in range(w):
                    for y in range(h):
                        template_h[y, x] = (sin_x*x/w + cos_x*y/h) ** 3
                min_value, max_value = np.min(template_h), np.max(template_h)
                template_h = (template_h - min_value) / (max_value - min_value)
            elif temp_type == 'sinwave':
                frequency = random.choice([0.5, 1, 1.5, 2, 3])
                theta = random.choice([0, 30, 60, 90])
                sin_x = math.sin(angle) ** 2 * (2 * (math.sin(angle) > 0) - 1)
                cos_x = math.cos(angle) ** 2 * (2 * (math.cos(angle) > 0) - 1)
                for x in range(w):
                    for y in range(h):
                        template_h[y, x] = (math.sin((sin_x*x/w + cos_x*y/h) * frequency * np.pi +
                                                     theta * np.pi / 180.0) + 1) / 2
                min_value, max_value = np.min(template_h), np.max(template_h)
                template_h = (template_h - min_value) / (max_value - min_value)
            template_h_list.append(template_h)

        return np.mean(np.dstack(template_h_list), axis=2, keepdims=True)

    def process(self, image):
        if isinstance(image, Image.Image):
            image = np.asarray(image)
        elif isinstance(image, np.ndarray):
            image = copy.deepcopy(image)
        else:
            assert False, 'False image type: {}, should be `mx.nd.NDArray` or `np.ndarray`.'.format(
                type(image))

        h, w = image.shape[:2]
        template_h = self.generate_template(h, w).reshape((h, w, 1))
        template_r = np.broadcast_to(
            template_h, (template_h.shape[0], template_h.shape[1], image.shape[2]))
        c = random.uniform(-self.brightness, self.brightness)
        image = image * (1 + template_r * c)

        return Image.fromarray(image.astype('uint8'))

    def __call__(self, img):
        # if True:
        #     if not os.path.exists('before_aug.jpg'):
        #         img.save('before_aug.jpg')

        image = self.process(img)

        # if True:
        #     if not os.path.exists('after_aug.jpg'):
        #         image.save('after_aug.jpg')

        return image


def _gaussian_blur(img, kernel_size_min, kernel_size_max, sigma_min, sigma_max):
    k = np.random.randint(kernel_size_min, kernel_size_max)
    if k % 2 == 0:
        if np.random.rand() > 0.5:
            k += 1
        else:
            k -= 1
    s = np.random.uniform(sigma_min, sigma_max)
    img_blur = cv2.GaussianBlur(src=img, ksize=(k, k), sigmaX=s)

    return img_blur


class GaussianBlur(object):
    def __init__(self, kernel_size_min, kernel_size_max, sigma_min, sigma_max):
        self.kernel_size_min = kernel_size_min
        self.kernel_size_max = kernel_size_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img, label=None):
        # if True:
        #     if not os.path.exists('before_aug.jpg'):
        #         img.save('before_aug.jpg')

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        img = _gaussian_blur(img, self.kernel_size_min, self.kernel_size_max,
                             self.sigma_min, self.sigma_max)
        img = Image.fromarray(img.astype('uint8'))

        # if True:
        #     if not os.path.exists('after_aug.jpg'):
        #         img.save('after_aug.jpg')

        return img


def _motion_blur(img, length_min, length_max, angle_min, angle_max):
    length = np.random.randint(length_min, length_max)
    angle = np.random.randint(angle_min, angle_max)

    if angle in [0, 90, 180, 270, 360]:
        angle += 1

    half = length / 2
    EPS = np.finfo(float).eps

    alpha = (angle - math.floor(angle / 180) * 180) / 180 * math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)

    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1

    # blur kernel size
    sx = int(math.fabs(length * cosalpha + psfwdt * xsign - length * EPS))
    sy = int(math.fabs(length * sinalpha + psfwdt - length * EPS))
    psf1 = np.zeros((sy, sx))

    # psf1 is getting small when (x, y) move from left-top to right-bottom
    # at this moment (x, y) is moving from right-bottom to left-top
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i * math.fabs(cosalpha) - j * sinalpha
            rad = math.sqrt(i*i + j*j)

            if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j] * sinalpha) / cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j] * psf1[i][j] + temp*temp)

            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])

            if psf1[i][j] < 0:
                psf1[i][j] = 0

    # anchor is (0, 0) when (x, y) is moving towards left-top
    anchor = (0, 0)
    # anchor is (width, heigth) when (x, y) is moving towards right-top
    if angle < 90 and angle > 0:  # flip kernel at this moment
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, 0)
    elif angle > -90 and angle < 0:  # moving towards right-bottom
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1] - 1, psf1.shape[0] - 1)
    elif angle < -90:  # moving towards left-bottom
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
    psf1 = psf1 / psf1.sum()

    img_blur = cv2.filter2D(src=img, ddepth=-1, kernel=psf1, anchor=anchor)

    return img_blur


class MotionBlur(object):
    def __init__(self, length_min, length_max, angle_min, angle_max):
        self.length_min = length_min
        self.length_max = length_max
        self.angle_min = angle_min
        self.angle_max = angle_max

    def __call__(self, img, label=None):
        if True:
            if not os.path.exists('before_aug.jpg'):
                img.save('before_aug.jpg')

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        img = _motion_blur(img, self.length_min, self.length_max,
                           self.angle_min, self.angle_max)
        img = Image.fromarray(img.astype('uint8'))

        if True:
            if not os.path.exists('after_aug.jpg'):
                img.save('after_aug.jpg')

        return img

