import os
import sys
import cv2
import numpy as np
from utils import *


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1])
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0])
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


def center2fov(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], np.float32)
    K_inv = np.linalg.inv(K)

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz)
    return lonlat


def get_tile_in_FoV(viewport, tile_h=args.tile_h, tile_w=args.tile_w,
                    h=args.sal_h, w=args.sal_w):
    (theta, phi) = viewport
    lonlat = center2fov(args.fov_span, theta, phi, h, w)
    fov_pos = lonlat2XY(lonlat, shape=(tile_h, tile_w)).astype(np.int)
    tile_in_fov = np.zeros((tile_h, tile_w))
    for i in range(h):
        for j in range(w):
            tile_in_fov[fov_pos[i, j, 1]][fov_pos[i, j, 0]] = 1
    return tile_in_fov


class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        lonlat = center2fov(FOV, THETA, PHI, height, width)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp


class Pixel2FoV:
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.fov_pos = {}
        self.pixel_weight = {}
        for x in range(h):
            for y in range(w):
                self.fov_pos[(x, y)], self.pixel_weight[(x, y)] = self.get_fov_pos(x, y)
        print('fov pos initializing finished')


    def get_fov_pos(self, x, y):
        lonlat0 = xy2lonlat(x, y, self.h, self.w)
        (theta, phi) = lonlat0
        lonlat = center2fov(args.fov_span, theta, phi, self.h, self.w)
        fov_pos = lonlat2XY(lonlat, shape=(self.h, self.w)).astype(np.float32)
        pixel_weight = np.zeros((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                lonlatxy = xy2lonlat(fov_pos[i, j, 1], fov_pos[i, j, 0], self.h, self.w)
                deg_distance = get_deg_distance(lonlatxy, lonlat0)
                pixel_weight[i, j] = gaussian_from_distance(deg_distance / 5.0)
        return fov_pos, pixel_weight

