import os
import pickle
from typing import Optional, List

import numpy as np
import py360convert
from matplotlib import pyplot as plt

from .e2plabelconvert import generateOnePerspectiveLabel, VIEW_ARGS, VIEW_NAME, VIEW_SIZE

data_dir = "../PanoHough/data/layoutnet_dataset/train"

COLORS = {
    0: "red",  # 竖直墙壁线红色
    1: "green",  # 天花板线绿色
    2: "blue",  # 地板线蓝色
}


def clearAxesLines(ax: plt.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


DRAW_CUBE_POSITIONS = {
    "F": [1, 1],
    "R": [1, 2],
    "B": [1, 3],
    "L": [1, 0],
    "U": [0, 1],
    "D": [2, 1],
    "E": [0, 1, 2, 4],
}


def getCubeAxes(fig: plt.Figure, view_name):
    spec = fig.add_gridspec(3, 4, hspace=0, wspace=0)
    posi = DRAW_CUBE_POSITIONS[view_name]
    if len(posi) == 2:
        ax: plt.Axes = fig.add_subplot(spec[posi[0], posi[1]])
    else:
        ax: plt.Axes = fig.add_subplot(spec[posi[0]:posi[1], posi[2]:posi[3]])
    clearAxesLines(ax)
    return ax


def drawOnePerspectivePreview(axs: Optional[List[plt.Axes]], p_img, r):
    points = r["points"]
    point_types = r["point_types"]
    lines = r["lines"]
    for ax in (axs if axs is not None else []):
        ax.imshow(p_img)
        ax.scatter(points[point_types == 1, 0], points[point_types == 1, 1])  # 画点只画图片范围内的
        for l in lines:
            ax.plot(l[[3, 5]], l[[4, 6]], color=COLORS[int(l[2])])
        ax.set_xlim(0, p_img.shape[1])
        ax.set_ylim(p_img.shape[0], 0)


def main(data_dir):
    SHOW_ORIGIN = False
    SAVE_PREVIEW_P = True
    SAVE_PERSPECTIVE_IMG_IN_LABEL_FILE = True

    img_dir = os.path.join(data_dir, "img")
    label_cor_dir = os.path.join(data_dir, "label_cor")
    preview_p_dir = os.path.join(data_dir, "preview_p")
    cube_view_dir = os.path.join(preview_p_dir, "cube")
    label_p_dir = os.path.join(data_dir, "label_p")
    # 创建文件夹
    os.makedirs(label_p_dir, exist_ok=True)
    for name in VIEW_NAME:
        os.makedirs(os.path.join(preview_p_dir, name), exist_ok=True)
    os.makedirs(cube_view_dir, exist_ok=True)

    names = []
    for i, name in enumerate(os.listdir(img_dir)):
        img_name = os.path.join(img_dir, name)
        img = plt.imread(img_name)
        basename = os.path.splitext(name)[0]
        label_cor_name = os.path.join(label_cor_dir, basename + ".txt")
        with open(label_cor_name, "r") as label_fileobj:
            label = np.array([[int(t) for t in s.split(" ")] for s in label_fileobj.readlines()])

        cube_view_fig = plt.Figure()
        cube_ax = getCubeAxes(cube_view_fig, "E")
        cube_ax.imshow(img)
        cube_ax.scatter(label[:, 0], label[:, 1])

        if SHOW_ORIGIN:
            plt.imshow(img)
            plt.scatter(label[:, 0], label[:, 1])
            plt.show()

        result = []
        for view_idx, (view_name, view) in enumerate(zip(VIEW_NAME, VIEW_ARGS)):
            img_save_path = os.path.join(preview_p_dir, view_name, basename + ".png")
            p_ax = [getCubeAxes(cube_view_fig, view_name)]
            if SAVE_PREVIEW_P:
                p_ax.append(plt.gca())
            p_img = py360convert.e2p(img, *view, VIEW_SIZE)
            r = generateOnePerspectiveLabel(img, label, *view, VIEW_SIZE)
            drawOnePerspectivePreview(p_ax, p_img, r)
            r["name"] = view_name
            if SAVE_PERSPECTIVE_IMG_IN_LABEL_FILE:
                r["img"] = p_img
            result.append(r)
            if SAVE_PREVIEW_P:
                plt.savefig(img_save_path)
                plt.clf()

        cube_view_fig.savefig(os.path.join(cube_view_dir, basename + ".jpg"))
        plt.close(cube_view_fig)

        with open(os.path.join(label_p_dir, basename + ".pkl"), "wb") as f:
            pickle.dump(result, f)
        print("%d %s" % (i, img_name))


if __name__ == '__main__':
    main(data_dir)
