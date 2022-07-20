import os
import warnings

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from easydict import EasyDict
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import transforms

from e2plabel.e2plabelconvert import generatePerspective, linesPostProcess
from misc import panostretch
from misc import post_proc


class PerspectiveDataset(data.Dataset):
    def __init__(self,
                 cfg: EasyDict,
                 split: str,
                 filename=None,
                 train_mode=False):
        self.cfg = cfg
        self.train_mode = train_mode
        self.rotate = None  # rotate # TODO 回原版代码核对一下 rotate怎么实现的？

        self.H, self.W = (512, 1024) if "IMG_SIZE" not in cfg.DATA else cfg.DATA.IMG_SIZE
        self.FOV = 90 if "FOV" not in cfg.DATA else cfg.DATA.FOV
        self.P = 512 if "PERSPECTIVE_SIZE" not in cfg.DATA else cfg.DATA.PERSPECTIVE_SIZE
        self.bin_num = 512
        self.hough_label_gradual_type = "exp"

        # e2p参数设定 详见e2plabelconvert.py
        self.view_args = [
            [(self.FOV, self.FOV), 0, 0],
            [(self.FOV, self.FOV), 90, 0],
            [(self.FOV, self.FOV), 180, 0],
            [(self.FOV, self.FOV), -90, 0],
            [(self.FOV, self.FOV), 0, 90],
            [(self.FOV, self.FOV), 0, -90],
        ]
        self.view_name = ['F', 'R', 'B', 'L', 'U', 'D']
        self.view_size = (self.P, self.P)

        self.ch = -1.6

        self.randomEraser = transforms.RandomErasing()

        # self._check_dataset()
        root_dir = cfg.DATA.ROOT_DIR
        self.path = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.path, 'img')
        self.cor_dir = os.path.join(self.path, 'label_cor')

        self.img_fnames = sorted([fname for fname in os.listdir(self.img_dir)])
        if cfg.DATA.get("PREFIX") is not None:
            self.img_fnames = [fname for fname in self.img_fnames
                               if sum([1 if fname.find(p) == 0 else 0 for p in cfg.DATA.PREFIX]) > 0  # 确保图片符合某一prefix
                               ]

        if filename is not None:  # 只使用单一的一张图片的情况
            self.img_fnames = [filename]

        # 读取每个图片的点label数据，预保存在内存里从而进行二次过滤
        self.cors = []
        for filename in self.img_fnames:
            with open(os.path.join(self.cor_dir, filename[:-4] + ".txt")) as f:
                cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)
                self.cors.append(cor)

        # 如果配置中对角点的个数有要求，执行过滤
        if cfg.DATA.get("USE_CORNER"):
            corner_count = [len(cor) // 2 for cor in self.cors]
            mask = [count in cfg.DATA.USE_CORNER for count in corner_count]
            self.img_fnames = [v for v, m in zip(self.img_fnames, mask) if m]
            self.cors = [v for v, m in zip(self.cors, mask) if m]

        # # TODO
        # try:
        #     idx = self.img_fnames.index("TbHJrupSAjP_235d08ff9f3f40ce9fa9e97696265dda.png")
        # except:
        #     idx = 0
        # self.img_fnames = self.img_fnames[idx:idx + 1] * 1000
        # self.cors = self.cors[idx: idx + 1] * 1000
        # a = 1

    def _check_dataset(self):
        for fname in self.txt_fnames:
            assert os.path.isfile(os.path.join(self.cor_dir, fname)), \
                '%s not found' % os.path.join(self.cor_dir, fname)
        # for fname in self.pkl_fnames:
        #     assert os.path.isfile(os.path.join(self.label_p_dir, fname)), \
        #         '%s not found' % os.path.join(self.label_p_dir, fname)

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self, idx):
        return self.getItem(self.img_fnames[idx], self.cors[idx])

    def getItem(self, filename, cor=None):
        # TODO 当前的实现是基于动态由e_img生成label和六个面的的信息的
        # 读取原图、角点label数据
        img_path = os.path.join(self.img_dir, filename)
        e_img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.

        if cor is None:
            with open(os.path.join(self.cor_dir, filename[:-4] + ".txt")) as f:
                cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

        # fname = self.img_fnames[idx]
        # P = self.P
        # l = np.tan(np.deg2rad(self.FOV / 2))
        # bin_num = self.bin_num
        H = self.H
        W = self.W

        # Use cor make smooth angle label
        # Corner with minimum x should at the beginning
        cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)
        # # Detect occlusion
        # occlusion = find_occlusion(cor[::2].copy()).repeat(2)

        AUG_RECORD = filename + " "
        # 数据增强
        # 只有train_mode才开数据增强
        if self.train_mode or self.cfg.get("TEST_NEED_AUG", False):
            # Stretch augmentation（把图片、label均进行缩放）
            if self.cfg.DATA.AUGMENT.get("stretch"):
                max_stretch = self.cfg.DATA.AUGMENT.stretch
                if max_stretch == True: max_stretch = 2.0  # 默认值
                xmin, ymin, xmax, ymax = cor2xybound(cor)
                kx = np.random.uniform(0.5, max_stretch)
                ky = np.random.uniform(0.5, max_stretch)
                a = np.random.randint(2)
                b = np.random.randint(2)
                if a == 0:
                    kx = max(1 / kx, min(0.5 / xmin, 1.0))
                else:
                    kx = min(kx, max(10.0 / xmax, 1.0))
                if b == 0:
                    ky = max(1 / ky, min(0.5 / ymin, 1.0))
                else:
                    ky = min(ky, max(10.0 / ymax, 1.0))
                e_img, cor, _ = panostretch.pano_stretch(e_img, cor, kx, ky)
                AUG_RECORD += "estre{:f}{:f}{:d}{:d} ".format(kx, ky, a, b)

            # Random flip
            if self.cfg.DATA.AUGMENT.get("flip") and np.random.randint(2) == 0:
                e_img = np.flip(e_img, axis=1).copy()
                cor[:, 0] = e_img.shape[1] - 1 - cor[:, 0]
                AUG_RECORD += "efilp "

            # Random erase in random position
            if self.cfg.DATA.AUGMENT.get("erase") and np.random.randint(
                    self.cfg.DATA.AUGMENT.get("erase_EVERY", 2)) == 0:
                # H, W = e_img.shape[:2]
                n_holes = np.random.randint(self.cfg.DATA.AUGMENT.get("erase_COUNT", 10))
                hole_length_y = self.cfg.DATA.AUGMENT.get("erase_SIZE", 50)
                hole_length_x = self.cfg.DATA.AUGMENT.get("erase_SIZE", 50)
                mask = np.ones((H, W, 3), np.float32)
                noise = np.zeros((H, W, 3), np.float32)
                for n in range(n_holes):
                    xhole = np.random.randint(W)
                    yhole = np.random.randint(H)

                    yhole1 = np.clip(yhole - hole_length_y // 2, 0, H)
                    yhole2 = np.clip(yhole + hole_length_y // 2, 0, H)
                    xhole1 = np.clip(xhole - hole_length_x // 2, 0, W)
                    xhole2 = np.clip(xhole + hole_length_x // 2, 0, W)

                    mask[yhole1:yhole2, xhole1:xhole2] = 0
                    noise[yhole1:yhole2, xhole1:xhole2] = np.random.rand(yhole2 - yhole1, xhole2 - xhole1, 3)
                e_img = e_img * mask  # + noise

            if self.cfg.DATA.AUGMENT.get("bon_erase"):
                # H, W = img.shape[:2]
                n_holes = self.cfg.DATA.AUGMENT.get("erase_COUNT", 10)  # 10
                hole_length_y = self.cfg.DATA.AUGMENT.get("erase_SIZE", 50)  # 50
                hole_length_x = self.cfg.DATA.AUGMENT.get("erase_SIZE_X", 100)  # 100
                mask = np.ones((H, W, 3), np.float32)
                noise = np.zeros((H, W, 3), np.float32)
                bon_floor_x, bon_floor_y = cor[1::2, 0], cor[1::2, 1]
                bon_ceil_x, bon_ceil_y = cor[0::2, 0], cor[0::2, 1]
                bon_floor = np.interp(np.arange(W),
                                      bon_floor_x,
                                      bon_floor_y,
                                      period=W)
                bon_ceil = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
                for n in range(n_holes):
                    xhole = np.random.randint(W)
                    if True:  # self.bon_erase:
                        if n % 2 == 0:
                            yhole = int(bon_floor[xhole])
                        else:
                            yhole = int(bon_ceil[xhole])
                    else:  # if self.erase:
                        yhole = np.random.randint(H)
                    yhole1 = np.clip(yhole - hole_length_y // 2, 0, H)
                    yhole2 = np.clip(yhole + hole_length_y // 2, 0, H)
                    xhole1 = np.clip(xhole - hole_length_x // 2, 0, W)
                    xhole2 = np.clip(xhole + hole_length_x // 2, 0, W)
                    mask[yhole1:yhole2, xhole1:xhole2] = 0
                    noise[yhole1:yhole2,
                    xhole1:xhole2] = np.random.rand(yhole2 - yhole1,
                                                    xhole2 - xhole1, 3)
                e_img = e_img * mask + noise

            # Random gamma augmentation
            if self.cfg.DATA.AUGMENT.get("gamma"):
                p = np.random.uniform(1, 2)
                if np.random.randint(2) == 0:
                    p = 1 / p
                e_img = e_img ** p

            # Random noise augmentation
            if self.cfg.DATA.AUGMENT.get("noise"):
                if np.random.randint(2) == 0:
                    noise = np.random.randn(*e_img.shape) * 0.05
                    e_img = np.clip(e_img + noise, 0, 1)

        # TODO 把数据集存起来
        # save_dir = "processed_input/4_stf"
        # os.makedirs(save_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(save_dir, filename),
        #             cv2.cvtColor(np.round(e_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # 到此e_img的数据增强结束，开始生成gt使用的数据
        pres = generatePerspective(e_img, cor, self.view_name, self.view_args, self.view_size)
        # 解析pres的内容 转换为label向量
        p_imgs = []
        xLabels, yLabels, cUpLabels, cDownLabels = [], [], [], []
        peakss = []
        liness = []
        for d in pres:
            p_img = torch.FloatTensor(d["img"].transpose([2, 0, 1]))
            if self.train_mode or self.cfg.get("TEST_NEED_AUG", False):
                AUG_RECORD += d["name"] + " "
                # Stretch augmentation（把图片、label均进行缩放）
                if self.cfg.DATA.get("PERSPECTIVE_AUGMENT", {}).get("stretch") and np.random.randint(2) == 0:
                    max_stretch = self.cfg.DATA.PERSPECTIVE_AUGMENT.stretch
                    if max_stretch == True: max_stretch = 1.5  # 默认值
                    stretch_value = np.random.uniform(1.0, max_stretch)
                    originSize = p_img.shape[1:]
                    newSize = [round(v * stretch_value) for v in originSize]
                    ratio = np.array([n / o for o, n in zip(originSize, newSize)])
                    centerPos = np.array([(v - 1) // 2 for v in originSize])
                    p_img = transforms.Resize(newSize)(p_img)
                    p_img = transforms.CenterCrop(originSize)(p_img)
                    for line in d["lines"]:
                        line[3:5] = (line[3:5] - centerPos) * ratio + centerPos
                        line[5:7] = (line[5:7] - centerPos) * ratio + centerPos
                    AUG_RECORD += "stre{:f} ".format(stretch_value)

                # Random flip
                if self.cfg.DATA.get("PERSPECTIVE_AUGMENT", {}).get("vertical_flip") and np.random.randint(3) == 0:
                    p_img = p_img.flip(-2)
                    for line in d["lines"]:
                        line[4:7:2] = p_img.shape[-2] - 1 - line[4:7:2]
                    AUG_RECORD += "vert "

                if self.cfg.DATA.get("PERSPECTIVE_AUGMENT", {}).get("horizontal_flip") and np.random.randint(3) == 0:
                    p_img = p_img.flip(-1)
                    for line in d["lines"]:
                        line[3:7:2] = p_img.shape[-1] - 1 - line[3:7:2]
                    AUG_RECORD += "hori "

                rotated = False
                if self.cfg.DATA.get("PERSPECTIVE_AUGMENT", {}).get("rotation"):
                    r = np.random.randint(6)
                    if r == 0:
                        # 顺时针转90度
                        # 先转置，再水平翻转
                        p_img = p_img.transpose(-2, -1).flip(-1)
                        for line in d["lines"]:
                            line[3:7] = line[[4, 3, 6, 5]]
                            line[3:7:2] = p_img.shape[-1] - 1 - line[3:7:2]
                        rotated = True
                        AUG_RECORD += "rota0 "
                    elif r == 1:
                        # 逆时针转90度
                        # 先转置，再竖直翻转
                        p_img = p_img.transpose(-2, -1).flip(-2)
                        for line in d["lines"]:
                            line[3:7] = line[[4, 3, 6, 5]]
                            line[4:7:2] = p_img.shape[-2] - 1 - line[4:7:2]
                        rotated = True
                        AUG_RECORD += "rota1 "

                # Random erase in random position
                if self.cfg.DATA.get("PERSPECTIVE_AUGMENT", {}).get("erase"):
                    p_img = self.randomEraser(p_img)
                    # label不会因为erasing而改变

                # Random gamma augmentation
                if self.cfg.DATA.get("PERSPECTIVE_AUGMENT", {}).get("gamma"):
                    p = np.random.uniform(1, 2)
                    if np.random.randint(2) == 0:
                        p = 1 / p
                    p_img = p_img ** p

                # Random noise augmentation
                if self.cfg.DATA.get("PERSPECTIVE_AUGMENT", {}).get("noise") and np.random.randint(2) == 0:
                    noise = torch.randn(p_img.shape) * 0.025
                    p_img = torch.clip(p_img + noise, 0, 1)

                oldDirection = [vec[7] for vec in d["lines"]]
                d["lines"], mask = linesPostProcess(d["lines"], p_img.shape[1:], d["name"] == "U" or d["name"] == "D",
                                                    return_mask=True)
                oldDirection = [v for v, b in zip(oldDirection, mask) if b]
                if rotated and not (d["name"] == "U" or d["name"] == "D"):
                    # 当中间面发生了旋转时，需要根据oldDirection，原来是0的现在是1（计算结果为0），原来是1的现在是0（计算结果为2）
                    for i, (oldValue, newValue) in enumerate(zip(oldDirection, [vec[7] for vec in d["lines"]])):
                        if oldValue == 0:
                            assert newValue == 0
                            d["lines"][i][7] = 1
                        elif oldValue == 1:
                            # assert newValue == 2 # 这里并不需要，因为如果是绝对竖直线，yLR=nan，照样会输出direct=1.
                            assert newValue != 0  # 所以newValue也可能是1而不是2。只断言不是0就好
                            d["lines"][i][7] = 0

            p_imgs.append(p_img)
            if self.cfg.MODEL.HOUGH.CLINE_TYPE == "NEW":
                liness.append(np.array(d["lines"]))
                try:
                    peaks = self.linesToPeaksNew(d["lines"], self.view_size)
                except:
                    assert False, AUG_RECORD
                peakss.append(peaks)
                peaks_for_label = peaks
                label_hw = self.view_size
                xPeaks, yPeaks, cUpPeaks, cDownPeaks = peaks_for_label
                xLabels.append(
                    self.generate_gradual_hough_label(xPeaks, label_hw[1], type=self.hough_label_gradual_type,
                                                      base=self.cfg.MODEL.HOUGH.GRADUAL_LABEL.XY))
                yLabels.append(
                    self.generate_gradual_hough_label(yPeaks, label_hw[0], type=self.hough_label_gradual_type,
                                                      base=self.cfg.MODEL.HOUGH.GRADUAL_LABEL.XY))

                cline_angle_num = label_hw[1] + label_hw[0] // 2 * 2 - 2
                cUpLabels.append(
                    self.generate_gradual_hough_label(cUpPeaks, cline_angle_num, type=self.hough_label_gradual_type,
                                                      base=self.cfg.MODEL.HOUGH.GRADUAL_LABEL.CUPDOWN))
                cDownLabels.append(
                    self.generate_gradual_hough_label(cDownPeaks, cline_angle_num, type=self.hough_label_gradual_type,
                                                      base=self.cfg.MODEL.HOUGH.GRADUAL_LABEL.CUPDOWN))
            else:
                raise NotImplementedError()

        n_cor = len(cor)
        gt_floor_coor = cor[1::2]
        gt_ceil_coor = cor[0::2]
        gt_floor_xyz = np.hstack([
            post_proc.np_coor2xy(gt_floor_coor, self.ch, self.W, self.H, floorW=1, floorH=1),
            np.zeros((n_cor // 2, 1)) + self.ch,
        ])
        gt_c = np.sqrt((gt_floor_xyz[:, :2] ** 2).sum(1))
        gt_v2 = post_proc.np_coory2v(gt_ceil_coor[:, 1], self.H)
        gt_ceil_z = gt_c * np.tan(gt_v2)

        height = np.array([gt_ceil_z.mean() - self.ch], dtype=np.float32)

        # Convert all data to tensor
        e_img = torch.FloatTensor(e_img.transpose([2, 0, 1]))
        # angle = torch.FloatTensor(angle)
        # up_bin256 = torch.FloatTensor(up_bin256.copy())
        # down_bin256 = torch.FloatTensor(down_bin256.copy())
        height = torch.FloatTensor(height)

        out_dict = {
            "filename": filename,
            "e_img": e_img,
            "cor": cor,
            "height": height,
            "p_imgs": torch.stack(p_imgs, 0),
            "xLabels": np.array(xLabels).astype(np.float32),
            "yLabels": np.array(yLabels).astype(np.float32),
            "cUpLabels": np.array(cUpLabels).astype(np.float32),
            "cDownLabels": np.array(cDownLabels).astype(np.float32),
            "peaks": peakss,
            "lines": liness
        }
        return out_dict

    @staticmethod
    def generate_gradual_hough_label(peaks, res_len, loop=False, type="exp", base=0.96):
        """
        根据若干个峰值点，生成渐变的数组，越靠近峰值点值越大，从而用于网络的直接计算loss。
        :param peaks 数组，各个峰值点
        :param res_len 结果数组的长度
        :param loop 计算结果距离的时候是否视为是一个循环
        """
        res = []
        res.append(cdist(peaks.reshape(-1, 1), np.arange(res_len, dtype=np.float).reshape(-1, 1), p=1))
        if loop:
            res.append(cdist(peaks.reshape(-1, 1), np.arange(res_len).reshape(-1, 1) + res_len, p=1))
            res.append(cdist(peaks.reshape(-1, 1), np.arange(res_len).reshape(-1, 1) - res_len, p=1))
        dist = np.min(res, 0)
        if dist.shape[0] > 0:
            nearest_dist = dist.min(0)  # shape(res_len)，每个点距离最近的peak的距离
        else:
            # TODO 对于没有peak的情况要怎么处理？当作距离是inf是否合理？这样label就是0了
            nearest_dist = np.ones(dist.shape[1:], dtype=dist.dtype) * np.inf
        if type == "exp":
            return (base ** nearest_dist).reshape(-1)
        elif type == "nearest_only":
            return (nearest_dist.reshape(-1) <= 0.5).astype(nearest_dist.dtype)
        elif type == "nearest_k":
            return (nearest_dist.reshape(-1) <= base + 0.5).astype(nearest_dist.dtype)
        else:
            raise NotImplementedError()

    def linesToPeaks(self, lines, img_hw):
        """
        :return xPeaks, yPeaks, cUpPeaks（过中心的上半圈线）, cDownPeaks（过中心的下半圈线）
        """
        xPeaks, yPeaks, cUpPeaks, cDownPeaks = [], [], [], []
        for line in lines:
            if line[7] == 0:
                xPeaks.append(np.mean(line[3:7:2]))
            elif line[7] == 1:
                yPeaks.append(np.mean(line[4:7:2]))
            elif line[7] == 2:
                yCenter = np.mean(line[4:7:2])
                ks = (line[4:7:2] - ((img_hw[0] - 1) / 2)) / (line[3:7:2] - ((img_hw[1] - 1) / 2))
                # 角度规定为斜率的arctan。即上半圆，最左侧为0度、顺时针增长到180度；下半圆，最右侧为0度，顺时针增长到180度
                deg = np.rad2deg(np.arctan(ks))
                deg[deg < 0] += 180
                meanAngleDeg = np.mean(deg)
                if yCenter <= img_hw[0] / 2:
                    cUpPeaks.append(meanAngleDeg)
                else:
                    cDownPeaks.append(meanAngleDeg)
        return np.array(xPeaks), np.array(yPeaks), np.array(cUpPeaks), np.array(cDownPeaks)

    @staticmethod
    def coord2AngleValue(x, y, img_hw):
        """
        根据所属区域，求交线坐标，并直接转换为角度数量值
        :param x,y 直接输入图片中的坐标即可，不是中心坐标系
        :return 对应于new算法的angle值；0或1，表示上半图还是下半图
        """
        h2, w2 = (img_hw[0] - 1) / 2, (img_hw[1] - 1) / 2
        h2f = img_hw[0] // 2

        x = x - w2
        y = y - h2
        if x <= y <= -x:
            # 与左侧相交
            y2 = y / x * -w2
            if y <= 0:
                r = h2f - 1 - h2 - y2
                return r, 0
            else:
                r = h2f + img_hw[1] - 2 + h2 - y2
                return r, 1
        elif -x <= y <= x:
            # 与右侧相交
            y2 = y / x * w2
            if y <= 0:
                r = h2f + img_hw[1] - 2 + h2 + y2
                return r, 0
            else:
                r = h2f - 1 - h2 + y2
                return r, 1
        elif -y < x < y:
            # 与下侧相交
            x2 = x / y * h2
            r = h2f - 1 + w2 - x2
            return r, 1
        elif y < x < -y:
            # 与上侧相交
            x2 = x / y * -h2
            r = h2f - 1 + w2 + x2
            return r, 0

    @staticmethod
    def linesToPeaksNewCore(lines, img_hw):
        """
        :input: lines(n, 5) 五维分别代表x1,y1,x2,y2,线在视图中的类型-0竖直线1水平线2过中心线
        :return xPeaks, yPeaks, cUpPeaks（过中心的上半圈线）, cDownPeaks（过中心的下半圈线）
        """
        if isinstance(lines, list):
            lines = np.array(lines)

        def autoAbs(v):
            if isinstance(v, torch.Tensor):
                return v.abs()
            elif isinstance(v, np.ndarray):
                return np.abs(v)
            return abs(v)

        def toNdarrayOrTensor(v, ref):
            if isinstance(ref, torch.Tensor):
                return ref.new_tensor(v)
            else:
                return np.array(v)

        xPeaks, yPeaks, cUpPeaks, cDownPeaks = [], [], [], []
        xLengths, yLengths, cUpLengths, cDownLengths = [], [], [], []
        for line in lines:
            length_ratio = autoAbs(line[0:2] - line[2:4]) / toNdarrayOrTensor(img_hw, line)[[1, 0]]
            if line[4] == 0:
                xPeaks.append(line[0:4:2].mean())
                xLengths.append(length_ratio[1])
            elif line[4] == 1:
                yPeaks.append(line[1:4:2].mean())
                yLengths.append(length_ratio[0])
            elif line[4] == 2:
                # 对两个端点，计算其对应的角度（以边缘坐标系值为单位），两值直接算术平均作为最终的代表角度
                r1, p1 = PerspectiveDataset.coord2AngleValue(*line[0:2], img_hw)
                r2, p2 = PerspectiveDataset.coord2AngleValue(*line[2:4], img_hw)
                if p1 != p2:
                    warnings.warn("cline two endpoint is not in same updown part!")
                    midPointDis = np.abs(line[1::2] - ((img_hw[0] - 1) / 2))
                    if midPointDis.argmin() == 0:  # 应调整一号点
                        if p2 == 0:  # 2号点在上半图，1号点也放到上半图
                            line[1] = ((img_hw[0] - 1) / 2) - 0.01
                        else:
                            line[1] = ((img_hw[0] - 1) / 2) + 0.01
                    else:
                        if p1 == 0:
                            line[3] = ((img_hw[0] - 1) / 2) - 0.01
                        else:
                            line[3] = ((img_hw[0] - 1) / 2) + 0.01
                    r1, p1 = PerspectiveDataset.coord2AngleValue(*line[0:2], img_hw)
                    r2, p2 = PerspectiveDataset.coord2AngleValue(*line[2:4], img_hw)
                assert p1 == p2, "cline two endpoint is not in same updown part!"
                meanAngleValue = (r1 + r2) / 2  # 直接求算术平均
                if p1 == 0:
                    cUpPeaks.append(meanAngleValue)
                    cUpLengths.append(length_ratio.max() * 2)
                else:
                    cDownPeaks.append(meanAngleValue)
                    cDownLengths.append(length_ratio.max() * 2)
        return (xPeaks, yPeaks, cUpPeaks, cDownPeaks), (xLengths, yLengths, cUpLengths, cDownLengths)

    def linesToPeaksNew(self, lines, img_hw):
        return [np.array(item) for item in self.linesToPeaksNewCore([line[3:8] for line in lines], img_hw)[0]]

    @staticmethod
    def collate(batch):
        def collateByKey(batch, key):
            if key == "cor":
                return [PerspectiveDataset.collate(d[key]) for d in batch]
            else:
                return PerspectiveDataset.collate([d[key] for d in batch])

        elem = batch[0]
        if isinstance(elem, dict):
            return {key: collateByKey(batch, key) for key in elem}
        elif isinstance(elem, list) or isinstance(elem, tuple):
            return [PerspectiveDataset.collate(d) if isinstance(d[0], list) or isinstance(d[0], tuple)
                    else [default_collate([v]).squeeze(0) for v in d]
                    for d in batch]
        return default_collate(batch)


def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostretch.coorx2u(corU[:, 0])
    vU = panostretch.coory2v(corU[:, 1])
    vB = panostretch.coory2v(corB[:, 1])

    x, y = panostretch.uv2xy(u, vU, z=zU)
    c = np.sqrt(x ** 2 + y ** 2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)


if __name__ == '__main__':
    cfg = EasyDict()
    data = EasyDict()
    cfg.DATA = data
    cfg.DATA.ROOT_DIR = "data/layoutnet_dataset"
    dataset = PerspectiveDataset(cfg, "train")
    d = dataset[0]
    a = 1


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    seed = worker_info.seed
    np.random.seed((seed + _) % 2 ** 32)
    # # Avoid "cannot pickle KVReader object" error
    # dataset.reader = KVReader(dataset.path, dataset.num_readers)
