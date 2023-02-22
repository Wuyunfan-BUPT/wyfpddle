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
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
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
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
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
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
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
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
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
    print(getResult_W(g_folder, s_folder))
    print(getResult_T(g_folder, s_folder))
    print(getResult_E(g_folder, s_folder))

if __name__ == '__main__':
    s_folder = input()
    main(s_folder)
    #upper_swim
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult\model\swin\pretrained_upper\iter_120k_all'
    #s_folder = r'E:\bigpaper\swin_transformer_unet1_base\best_model_result\pseudo_color_prediction'
    #s_folder = r'E:\bigpaper\swin_transformer_unet1_small\iter_40000_result_8452\pseudo_color_prediction'
    # unet
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult\unet\unet_8203'

    # uper-swin
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult\P_upper_swin\iter_250k'
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult/newDatasetSwim/upernet_swin_zcrose/iter_230k'



    # uper_swin1
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/uper_swin1/uper_swin_deep_8312'

    # uper_swin2
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/uper_swin2/uper_swin2_8336'

    # uper_swin3
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/uper_swin3/uper_swin3_8009'


    # deeplav_res
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/deeplav_res/deeplav_res_8066'
    # unet-individual
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult/finaltest/fcn_unet_individual_P/iter_200k_P'


    # uper_res
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/uper_rest/up_res_81257'


    #s_folder = r'E:\bigpaper\unet_plus_plus\iter_40000_result_8329\pseudo_color_prediction'
    # s_folder = r'E:\bigpaper\unet3plus\iter_60000_result_8364\pseudo_color_prediction'
    #s_folder = r'E:\test\result_unet_240000_final\pseudo_color_prediction'
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult\finaltest\fcn_unet_individual_P\iter_500k'
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult\finaltest\fcn_vanunet_new\iter_300k'
    # g_folder = r'E:\GoogleDownload\braTS2020\test\annotations\test'
    # print(getResult_T(g_folder, s_folder))



    # fcn_swinconvunet
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult\deeplabv3_swinconvunet_plus\iter_290K'
    # g_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\data\bratsIndividual4C\annotations\test'
    # print(getResult_W(g_folder, s_folder))

    # deeplav3_swimconvunet


    # # fcn_swim
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult\model\swin\fcn_swin\iter_160K'
    # # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult\finaltest\fcn_vanunet_new\iter_300k'
    # g_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\data\bratsIndividual4C\annotations\test'
    # print(getResult_W(g_folder, s_folder))


