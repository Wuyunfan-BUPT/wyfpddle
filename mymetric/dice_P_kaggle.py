#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Project : Guangfa UI test
# @Time    : 2023/1/16 13:31
# @Author  : wuyfee
# @File    : dice.py
# @Software: PyCharm

from medpy import metric
import numpy as np
import os
import cv2
import argparse


def dice(res, ref):
    return metric.binary.dc(res, ref)

def getResult_W(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    dices=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_P = seg_img[seg_img < 255].sum()
        seg_img[seg_img < 255] = 1
        seg_img[seg_img == 255] = 0


        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}.png", cv2.IMREAD_UNCHANGED)
        if gt_img.sum() == 0 and seg_P==0:
            continue
        gt_img[gt_img>0] = 1

        dicet = dice(seg_img, gt_img)
        dices.append(dicet)

    dices = np.asanyarray(dices)

    print(f"dices: {dices.mean()}")
    return dices.mean()



def getResult_W_P(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    dices=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_P = seg_img
        seg_img[seg_img != 128] = 1
        seg_img[seg_img == 128] = 0

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}.png", cv2.IMREAD_UNCHANGED)
        if gt_img.sum() == 0 and seg_P.sum()==0:
            continue
        gt_img[gt_img>0] = 1

        dicet = dice(seg_img, gt_img)
        dices.append(dicet)

    dices = np.asanyarray(dices)

    print(f"dices: {dices.mean()}")
    return dices.mean()


def getResult_T(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    dices=[]

    for name in seg_names:
        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_P = seg_img[seg_img<255].sum()
        seg_img[seg_img < 64] = 1
        seg_img[seg_img > 64] = 0


        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}.png", cv2.IMREAD_UNCHANGED)
        if gt_img.sum() == 0 and seg_P==0:
            continue
        gt_img[gt_img==2] = 0
        gt_img[gt_img == 3] = 1

        dicet = dice(seg_img, gt_img)
        dices.append(dicet)

    dices = np.asanyarray(dices)
    print(f"dices: {dices.mean()}")
    return dices.mean()

def getResult_E(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    dices=[]

    for name in seg_names:



        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_P = seg_img[seg_img < 255].sum()
        seg_img[seg_img != 34] = 0
        seg_img[seg_img == 34] = 3

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}.png", cv2.IMREAD_UNCHANGED)
        if gt_img.sum() == 0 and seg_P==0:
            continue
        gt_img[gt_img != 3] = 0

        dicet = dice(seg_img, gt_img)
        dices.append(dicet)
    dices = np.asanyarray(dices)
    print(f"dices: {dices.mean()}")


    return dices.mean()


def main(s_folder):
    g_folder = r'data/zcross/annotations/test'
    print("Dice WT: ", getResult_W(g_folder, s_folder))
    print("Dice TC: ", getResult_T(g_folder, s_folder))
    print("Dice ET: ",getResult_E(g_folder, s_folder))

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--DirPath', default='None', required=True, help='请输入目录')
    args = parse.parse_args()
    s_folder = args.DirPath
    main(s_folder)
