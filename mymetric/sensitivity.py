#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Project : Guangfa UI test
# @Time    : 2023/1/17 0:43
# @Author  : wuyfee
# @File    : sensitivity.py
# @Software: PyCharm


from medpy import metric
import numpy as np
import os
import cv2

def sen(res, ref):
    return metric.binary.sensitivity(res, ref)

def getResult_W(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    sens=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img < 255] = 1
        seg_img[seg_img == 255] = 0

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img>0] = 1

        dicet = sen(seg_img, gt_img)
        sens.append(dicet)

    sens = np.asanyarray(sens)

    print(f"dices: {sens.mean()}")
    return sens.mean()


def getResult_T(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    sens=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img < 64] = 1
        seg_img[seg_img > 64] = 0


        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img==2] = 0
        gt_img[gt_img == 3] = 1

        dicet = sen(seg_img, gt_img)
        sens.append(dicet)

    sens = np.asanyarray(sens)
    print(f"dices: {sens.mean()}")
    return sens.mean()


def getResult_E(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    sens=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img != 34] = 0
        seg_img[seg_img == 34] = 3

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img!=3] = 0

        dicet = sen(seg_img, gt_img)
        sens.append(dicet)
    sens = np.asanyarray(sens)
    print(f"dices: {sens.mean()}")


    return sens.mean()


if __name__ == '__main__':

    s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult\model\swin\pretrained_upper\iter_120k_all'
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult\finaltest\fcn_unet_individual_P\iter_500k'
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult\finaltest\fcn_vanunet_new\iter_300k'
    g_folder = r'E:\GoogleDownload\braTS2020\test\annotations\test'
    print(getResult_T(g_folder, s_folder))