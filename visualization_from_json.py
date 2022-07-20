import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.figure import Figure
from tqdm import trange

from e2plabel.e2plabelconvert import generatePerspective, VIEW_NAME, VIEW_ARGS
from postprocess.postprocess2 import _cal_p_pred_emask
from visualization import clearAxesLines

img_hw = (512, 512)
e_img_hw = (512, 1024)


def jsonToCor(filename):
    H, W = e_img_hw
    with open(filename) as f:
        inferenced_result = json.load(f)
    cor_id = np.array(inferenced_result['uv'], np.float32)
    cor_id[:, 0] *= W
    cor_id[:, 1] *= H
    return cor_id


def txtToCor(filename):
    with open(filename) as f:
        cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)
    return cor


def resolveImgPath(s: str):
    if s.find("pano") == 0 or s.find("camera") == 0:
        return os.path.join("data/layoutnet_dataset/test/img", s)
    else:
        return os.path.join("data/matterport3d_layout/test/img", s)


def resolveGtCorPath(s: str):
    if s.find("pano") == 0 or s.find("camera") == 0:
        return os.path.join("data/layoutnet_dataset/test/label_cor", os.path.splitext(s)[0] + ".txt")
    else:
        return os.path.join("data/matterport3d_layout/test/label_cor", os.path.splitext(s)[0] + ".txt")


def corTo2DMask(e_img, cor):
    pres = generatePerspective(e_img, cor, VIEW_NAME, VIEW_ARGS, img_hw)
    lines = []
    for d in pres:
        lines.append(torch.tensor(d["lines"]))

    masks2d = []
    for view_idx in range(6):
        thickness = int(round(img_hw[0] * 0.01))
        mat = np.zeros((1, *img_hw))
        for line in lines[view_idx]:
            cv2.line(mat[0], torch.round(line[3:5]).to(torch.int64).numpy(),
                     torch.round(line[5:7]).to(torch.int64).numpy(), 1.0, thickness=thickness)
        masks2d.append(torch.tensor(mat))
    masks2d = torch.stack(masks2d)

    maskEq = _cal_p_pred_emask(None, masks2d, img_hw, e_img_hw)
    return maskEq


def wireframeGetMaskImg(e_img, cor, color) -> torch.Tensor:
    maskEq = corTo2DMask(e_img, cor).squeeze()
    mask_img = torch.cat([torch.tensor(color).repeat(*maskEq.shape[0:2], 1), maskEq.unsqueeze(-1)], 2)
    mask_img = torch.round(mask_img * 255).to(torch.uint8)
    return mask_img


def drawWireframeOnEImg(e_img, cor, color):
    plt.imshow(wireframeGetMaskImg(e_img, cor, color).cpu().numpy())


fig: Figure = None


def show(output_path, name):
    if output_path:
        plt.savefig(os.path.join(output_path, imgPath + "." + name + ".png"))
    else:
        plt.show()
    plt.close(fig)


def initFig():
    global fig
    fig = plt.figure(figsize=(10.24, 5.12))
    plt.gcf().subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    ax = plt.gca()
    clearAxesLines(ax)


CLASS_A = [
    "7y3sRwLe3Va_9b72664399a34e4f9dbe470571c73187.png",
    "B6ByNegPMKs_8b1abc1b47784d758b9ec1e079160475.png",
    "camera_1a2b3c7901434d88bba55d6f2b28a6d5_office_30_frame_equirectangular_domain_.png",
    "camera_7a42df17b40c4c15bfd6301823b6a476_office_22_frame_equirectangular_domain_.png",
    "camera_8cbbb3e42c0e4e54b3b523b1fec6b3bc_office_33_frame_equirectangular_domain_.png",
    "camera_412ba0d035b5432abd88ed447716f349_office_30_frame_equirectangular_domain_.png",
    "camera_514bd77b98cc47ad904d6c8196f769b1_office_8_frame_equirectangular_domain_.png",
    "camera_d162082c8f714aee8984195e0c5a7396_office_11_frame_equirectangular_domain_.png",
    "e9zR4mvMWw7_f624a40d100144e696a39abe258ee090.png",
    "pano_adxsvoaiehisue.png",
    "pano_agpqpoljoyzxds.png",
    "pano_ahvuapixtvirde.png",
    "pano_aixninerbhvojf.png",
    "pano_ankughvvgbhsom.png",
    "pano_apozlylyjgtjid.png",
    "uNb9QFRL6hY_1434b965c3c147419c4ff40310633b58.png",
    "x8F5xyUWy9e_2669f5ba693c4e729d7d2c4f3fa0a077.png",
]

CLASS_B = [
    "pano_aghlgnaxvjlzmb.png",
    "7y3sRwLe3Va_92fb09a83f8949619b9dc5bda2855456.png",
    "7y3sRwLe3Va_fdab6422162e49db822a37178ab70481.png",
    "B6ByNegPMKs_53249ef8a94c4c40bd6f09c069e54d16.png",
    "B6ByNegPMKs_bb2332e3d7ad40a59ee5ad0eae108dec.png",
    "B6ByNegPMKs_ce2f5a74556c4be192df3ca7a178cefb.png",
    "camera_32caf5752a4746c8b95f84e9acd9271d_office_29_frame_equirectangular_domain_.png",
    "camera_63eb2cd447b84c5abac846f79c51dfcd_office_14_frame_equirectangular_domain_.png",
    "camera_90af0a7fe0ed4a7db2c2e05727560231_office_15_frame_equirectangular_domain_.png",
    "camera_270448008f5743f48f34539d36e4c4ae_office_14_frame_equirectangular_domain_.png",
    "pano_auqcjiehbmenao.png",
    "wc2JMjhGNzB_6e491bc8576345bda3cdde9ab216b7be.png",
]

CLASS_C_D = [
    "7y3sRwLe3Va_9e4c92fd7eb74504baecf55a3264716e.png",
    "7y3sRwLe3Va_6376b741b50a4418b3dc3fde791c3c09.png",
    "B6ByNegPMKs_5b3d1c9fefb64512b0c9750a00feece4.png",
    "B6ByNegPMKs_e5567bd5fa2d4fde8a6b9f15e3274a7e.png",
    "e9zR4mvMWw7_5d711de78dbd400aa4cfd51fc05dfbee.png",
    "pano_abbvryjplnajxo.png",
    "pano_aqdafdzfhdukpg.png",
    "uNb9QFRL6hY_d11f14ddecbe406681d4980365ea5a43.png",
    "7y3sRwLe3Va_dd83fb40a2e14ac99de9fe9bcfaf44df.png",
    "uNb9QFRL6hY_bcce4f23c12744c782c0b49b24a0331a.png",
    "camera_a39f4a868cd84429a765324af21c6e6e_office_8_frame_equirectangular_domain_.png",
]

PANO_ARR = []
STF_ARR = []
MATTER_ARR = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', help='指定一张图片。如果不指定，那就会画所有')
    parser.add_argument('--output_path', help='如果不指定，就会plt.show')
    parser.add_argument('--draw_independent', "-d", action="store_true", help='独立画图还是一张图画好几次？')
    parser.add_argument('--draw_both', "-b", action="store_true", help='两种方法都画')
    parser.add_argument('--second', "-2", action="store_true", help='定义此项则画HoHoNet和AtlantaNet，否则画HorizonNet和LayoutNet')
    args = parser.parse_args()

    GT_COLOR = (0.0, 1.0, 0.0)  # 绿
    OUR_PATH, OUR_COLOR = "result_json", (1.0, 0.0, 0.0)  # 红
    HORIZONNET_PATH, HORIZONNET_COLOR = "eval_results/horizonnet_json", (0.0, 0.0, 1.0)  # 蓝
    LAYOUTNET_PATH, LAYOUTNET_COLOR = "eval_results/layoutnet_json", (1.0, 0.0, 1.0)  # 粉
    ATLANTANET_PATH, ATLANTANET_COLOR = "eval_results/atlantanet_json", (0.0, 0.0, 1.0)  # 蓝
    HOHONET_PATH, HOHONET_COLOR = "eval_results/hohonet_json", (1.0, 0.0, 1.0)  # 粉

    if args.img:
        img_list = [args.img]
    else:
        img_list = [s.replace(".json", "") for s in os.listdir(OUR_PATH)]

    # # TODO
    # img_list = CLASS_A + CLASS_B + CLASS_C_D
    # args.output_path = "result_6_pick"

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    for i in trange(len(img_list)):
        imgPath = img_list[i]
        isPanoStf = imgPath.find("pano") == 0 or imgPath.find("camera") == 0
        if args.second and isPanoStf: continue

        e_img = np.array(Image.open(resolveImgPath(imgPath))) / 255.0
        gt_cor = txtToCor(resolveGtCorPath(imgPath))
        myJsonPath = os.path.join(OUR_PATH, imgPath + ".json")
        with open(myJsonPath) as f:
            my_result = json.load(f)
            iou3d = my_result["3DIoU"]

            # # TODO
            # if imgPath.find("pano") == 0:
            #     PANO_ARR.append(iou3d)
            # elif imgPath.find("camera") == 0:
            #     STF_ARR.append(iou3d)
            # else:
            #     MATTER_ARR.append(iou3d)
            # continue

            # 画的顺序：gt、layout、horizon、ours
            if args.draw_both or (not args.draw_independent):
                initFig()
                plt.imshow(e_img)
                drawWireframeOnEImg(e_img, gt_cor, GT_COLOR)
                if not args.second:
                    drawWireframeOnEImg(e_img, txtToCor(os.path.join(LAYOUTNET_PATH, os.path.splitext(imgPath)[0] + (
                        "_aligned_rgb" if isPanoStf else "") + "_cor_id.txt")), LAYOUTNET_COLOR)
                    drawWireframeOnEImg(e_img, jsonToCor(os.path.join(HORIZONNET_PATH, os.path.splitext(imgPath)[0] + ".json")),
                                        HORIZONNET_COLOR)
                else:
                    drawWireframeOnEImg(e_img, jsonToCor(os.path.join(ATLANTANET_PATH, os.path.splitext(imgPath)[0] + ".json")),
                                        ATLANTANET_COLOR)
                    drawWireframeOnEImg(e_img, txtToCor(os.path.join(HOHONET_PATH, os.path.splitext(imgPath)[0] + ".layout.txt")),
                                        HOHONET_COLOR)
                drawWireframeOnEImg(e_img, jsonToCor(myJsonPath), OUR_COLOR)
                show(args.output_path, "all.{:.2f}".format(iou3d))
            if args.draw_both or args.draw_independent:
                if not args.second:
                    initFig()
                    plt.imshow(e_img)
                    drawWireframeOnEImg(e_img, gt_cor, GT_COLOR)
                    drawWireframeOnEImg(e_img, txtToCor(os.path.join(LAYOUTNET_PATH, os.path.splitext(imgPath)[0] + (
                        "_aligned_rgb" if isPanoStf else "") + "_cor_id.txt")), LAYOUTNET_COLOR)
                    show(args.output_path, "lay")

                    initFig()
                    plt.imshow(e_img)
                    drawWireframeOnEImg(e_img, gt_cor, GT_COLOR)
                    drawWireframeOnEImg(e_img, jsonToCor(os.path.join(HORIZONNET_PATH, os.path.splitext(imgPath)[0] + ".json")),
                                        HORIZONNET_COLOR)
                    show(args.output_path, "hor")

                else:
                    initFig()
                    plt.imshow(e_img)
                    drawWireframeOnEImg(e_img, gt_cor, GT_COLOR)
                    drawWireframeOnEImg(e_img, jsonToCor(os.path.join(ATLANTANET_PATH, os.path.splitext(imgPath)[0] + ".json")),
                                        ATLANTANET_COLOR)
                    show(args.output_path, "atl".format(iou3d))

                    initFig()
                    plt.imshow(e_img)
                    drawWireframeOnEImg(e_img, gt_cor, GT_COLOR)
                    drawWireframeOnEImg(e_img, txtToCor(os.path.join(HOHONET_PATH, os.path.splitext(imgPath)[0] + ".layout.txt")),
                                        HOHONET_COLOR)
                    show(args.output_path, "hoh".format(iou3d))

                initFig()
                plt.imshow(e_img)
                drawWireframeOnEImg(e_img, gt_cor, GT_COLOR)
                drawWireframeOnEImg(e_img, jsonToCor(myJsonPath), OUR_COLOR)
                show(args.output_path, "our.{:.2f}".format(iou3d))

        a = 1

    # TODO
    # import torch
    # PANO_ARR = torch.tensor(PANO_ARR).sort(descending=True)[0]
    # STF_ARR = torch.tensor(STF_ARR).sort(descending=True)[0]
    # MATTER_ARR = torch.tensor(MATTER_ARR).sort(descending=True)[0]
    # for a in [PANO_ARR,STF_ARR,MATTER_ARR]:
    #     pt = [round(len(a) / 4 * (i+1)) for i in range(3)]
    #     pt = [a[v] for v in pt]
    #     print(pt)
