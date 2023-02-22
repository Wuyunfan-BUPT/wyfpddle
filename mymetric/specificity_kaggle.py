#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Project : Guangfa UI test
# @Time    : 2023/1/17 0:45
# @Author  : wuyfee
# @File    : specificity.py
# @Software: PyCharm


from medpy import metric
import numpy as np
import os
import cv2
import argparse

def spe(res, ref):
    return metric.binary.specificity(res, ref)

def getResult_W(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    spes=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img < 255] = 1
        seg_img[seg_img == 255] = 0

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img>0] = 1

        dicet = spe(seg_img, gt_img)
        spes.append(dicet)

    spes = np.asanyarray(spes)

    print(f"dices: {spes.mean()}")
    return spes.mean()


def getResult_T(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    spes=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img < 64] = 1
        seg_img[seg_img > 64] = 0


        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img==2] = 0
        gt_img[gt_img == 3] = 1

        dicet = spe(seg_img, gt_img)
        spes.append(dicet)

    spes = np.asanyarray(spes)
    print(f"dices: {spes.mean()}")
    return spes.mean()


def getResult_E(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    spes=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img != 34] = 0
        seg_img[seg_img == 34] = 3

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img!=3] = 0

        dicet = spe(seg_img, gt_img)
        spes.append(dicet)
    spes = np.asanyarray(spes)
    print(f"dices: {spes.mean()}")


    return spes.mean()


def main(s_folder):
    g_folder = r'data/zcross/annotations/test'
    print("spe WT: ", getResult_W(g_folder, s_folder))
    print("spe TC: ", getResult_T(g_folder, s_folder))
    print("spe ET: ",getResult_E(g_folder, s_folder))

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--DirPath', default='None', required=True, help='请输入目录')
    args = parse.parse_args()
    s_folder = args.DirPath
    main(s_folder)