import cv2
import os
import sys


fname = sys.argv[1]
src_ddepth = sys.argv[2]

lines = open(fname).readlines()


def roi_scale(src_shape, dst_shape, roi):
    sw, sh = src_shape[:2]
    dw, dh = dst_shape[:2]

    scale_w = dw / sw
    scale_h = dh / sh

    sx1, sy1, sx2, sy2 = roi
    dx1 = sx1 * scale_w
    dx2 = sx2 * scale_w
    dy1 = sy1 * scale_h
    dy2 = sy2 * scale_h

    return dx1, dy1, dx2, dy2


for l in lines:
    path, *sroi, label = l.strip().split()
    im_depth = cv2.imread(os.path.join(src_ddepth, path))

    rgb_path = path.replace('depth', 'profile')
    im_rgb = cv2.imread(os.path.join(src_ddepth, rgb_path))

    roi = [float(x) for x in sroi]
    dst_depth_roi = roi_scale(im_rgb.shape, im_depth.shape, roi)
    ostr = '{} {} {}'.format(path, ' '.join([str(x) for x in dst_depth_roi]), label)
    print(ostr)

