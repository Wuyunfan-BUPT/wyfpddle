#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Project : Guangfa UI test
# @Time    : 2023/1/16 13:25
# @Author  : wuyfee
# @File    : Hausdorff.py
# @Software: PyCharm


import os
import sys
sys.path.append('./')
import numpy as np
from scipy import ndimage
import cv2
from hausdorff import hausdorff_distance



# def border_map(binary_img,neigh):
#     """
#     Creates the border for a 3D image
#     """
#     binary_map = np.asarray(binary_img, dtype=np.uint8)
#     neigh = neigh
#     west = ndimage.shift(binary_map, [-1, 0,0], order=0)
#     east = ndimage.shift(binary_map, [1, 0,0], order=0)
#     north = ndimage.shift(binary_map, [0, 1,0], order=0)
#     south = ndimage.shift(binary_map, [0, -1,0], order=0)
#     top = ndimage.shift(binary_map, [0, 0, 1], order=0)
#     bottom = ndimage.shift(binary_map, [0, 0, -1], order=0)
#     cumulative = west + east + north + south + top + bottom
#     border = ((cumulative < 6) * binary_map) == 1
#     return border
#
# def border_distance(ref,seg):
#     """
#     This functions determines the map of distance from the borders of the
#     segmentation and the reference and the border maps themselves
#     """
#     neigh=8
#     border_ref = border_map(ref,neigh)
#     border_seg = border_map(seg,neigh)
#     oppose_ref = 1 - ref
#     oppose_seg = 1 - seg
#     # euclidean distance transform
#     distance_ref = ndimage.distance_transform_edt(oppose_ref)
#     distance_seg = ndimage.distance_transform_edt(oppose_seg)
#     distance_border_seg = border_ref * distance_seg
#     distance_border_ref = border_seg * distance_ref
#     return distance_border_ref, distance_border_seg#, border_ref, border_seg
#
# def Hausdorff_distance(ref,seg):
#     """
#     This functions calculates the average symmetric distance and the
#     hausdorff distance between a segmentation and a reference image
#     :return: hausdorff distance and average symmetric distance
#     """
#     ref_border_dist, seg_border_dist = border_distance(ref,seg)
#     hausdorff_distance = np.max(
#         [np.max(ref_border_dist), np.max(seg_border_dist)])
#     return hausdorff_distance
#
# def hausdorff_whole (seg,ground):
#     return Hausdorff_distance(seg==0,ground==0)



# print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="euclidean") ))

def getResult_W(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    dices=[]

    for name in seg_names:

        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img < 255] = 1
        seg_img[seg_img == 255] = 0

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img>0] = 1
        if seg_img.max==0 and gt_img.max==0:
            dices.append(1)
        else:
            dicet = hausdorff_distance(seg_img, gt_img, distance="euclidean")
            if dicet >0:
                dices.append(dicet)
            else:
                dices.append(1)





    # values = np.asarray(values)



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
        seg_img[seg_img < 64] = 1
        seg_img[seg_img > 64] = 0


        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img==2] = 0
        gt_img[gt_img == 3] = 1
        # if seg_img.max==0 and gt_img.max==0:
        #     dices.append(1)
        # else:

        dicet = hausdorff_distance(seg_img, gt_img, distance="euclidean")
        dices.append(dicet)
            # if dicet >0:
            #     dices.append(dicet)
            # else:
            #     dices.append(1)





    # values = np.asarray(values)



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
        seg_img[seg_img != 34] = 0
        seg_img[seg_img == 34] = 3

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img!=3] = 0
        # if seg_img.max==0 and gt_img.max==0:
        #     dices.append(1)
        # else:

        dicet = hausdorff_distance(seg_img, gt_img, distance="euclidean")
        dices.append(dicet)
            # if dicet >0:
            #     dices.append(dicet)
            # else:
            #     dices.append(1)





    # values = np.asarray(values)



    dices = np.asanyarray(dices)




    print(f"dices: {dices.mean()}")


    return dices.mean()


if __name__ == '__main__':
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult\model\swin\pretrained_upper\iter_120k_all'
    # uper_swin
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult\P_upper_swin\iter_250k'

    # uper-swin
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult\P_upper_swin\iter_250k'
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult/newDatasetSwim/upernet_swin_zcrose/iter_230k'

    # uper_swin1
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/uper_swin1/uper_swin_deep_8312'

    # uper_swin2
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/uper_swin2/uper_swin2_8336'

    # uper_swin3
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/uper_swin3/uper_swin3_8009'

    # deeplav_res
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/deeplav_res/deeplav_res_8066'

    # uper_res
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/uper_rest/up_res_81257'

    # fpn_swin_deep
    s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult/fpn_swin_deep/fpn_swin_deep_8303'




    # unet
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\finalResult\unet\unet_8286'
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult\finaltest\fcn_unet_individual_P\iter_300k'
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult\finaltest\fcn_vanunet_new\iter_300k'
    g_folder = r'E:\GoogleDownload\braTS2020\test\annotations\test'
    print(getResult_T(g_folder, s_folder))

    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult\model\swinunet\fcn_swinconvunet\iter_300k'
    # g_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\data\bratsIndividual4C\annotations\test'
    # print(getResult_T(g_folder, s_folder))
    #

    # # deeplav3_swimconvunet
    # s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult\deeplabv3_swinconvunet_plus\iter_290K'
    # g_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\data\bratsIndividual4C\annotations\test'
    # print(getResult_W(g_folder, s_folder))


