#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Project : Guangfa UI test
# @Time    : 2023/1/15 19:22
# @Author  : wuyfee
# @File    : evaluation.py
# @Software: PyCharm

import numpy as np
import os
import cv2
from medpy import metric
import numpy as np


def sensitivity(seg,ground):
    #computs false negative rate
    num=np.sum(np.multiply(ground, seg))
    denom = 0
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if seg[i][j]==0 and ground[i][j]==1:
                denom = denom+1

    denom=denom + num
    if denom==0:
        return 1
    else:
        return num/denom

# def hausdorff95(res, ref):
#     return metric.binary.hd95(res, ref)

def sen(res, ref):
    return metric.binary.sensitivity(res, ref)

def spe(res, ref):
    return metric.binary.specificity(res, ref)
def recall(res, ref):
    return metric.binary.recall(res, ref)
def dice(res, ref):
    return metric.binary.dc(res, ref)
def voe(res, ref):
    return 1.-metric.binary.jc(res, ref)

from numpy.core.umath_tests import inner1d
def _haus_dist_95(A, B):
    """ compute the 95 percentile hausdorff distance """
    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
    dist1 = np.min(D_mat, axis=0)
    dist2 = np.min(D_mat, axis=1)
    hd95 = np.percentile(np.hstack((dist1, dist2)), 95)

    # hd = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))

    return hd95


# def hausdorff95(output, target):
#     with torch.no_grad():
#         # pred = torch.sigmoid(output) > 0.5
#
#         pred = output > 0.5
#         target = target > 0.5
#         assert pred.size() == target.size()
#
#         pred = pred.detach().cpu().numpy()
#         target = target.detach().cpu().numpy()
#
#         hd95 = 0.0
#         n = 0
#         for k in range(pred.shape[0]):
#             if np.count_nonzero(pred[k]) > 0:  # need to handle blank prediction
#                 n += 1
#                 pred_contours = pred[k] & (~morphology.binary_erosion(pred[k]))
#                 # print(type(target[k]))
#                 # import pdb;pdb.set_trace()
#                 target_contours = target[k] & (~morphology.binary_erosion(target[k]))
#                 pred_ind = np.argwhere(pred_contours)
#                 target_ind = np.argwhere(target_contours)
#                 hd95 += _haus_dist_95(pred_ind, target_ind)
#     return hd95, n
def getResult(gt_path, seg_path):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    hausdos=[]
    voes=[]
    dices=[]
    recalls=[]
    SENs=[]
    SPEs=[]
    for name in seg_names:

        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img < 255] = 1
        seg_img[seg_img == 255] = 0

        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img>0] = 1
        if gt_img.max()==0 and seg_img.max()==0:
            hausdos.append(0)
        else:
            hausdos.append(_haus_dist_95(seg_img, gt_img))
        # voes.append(voe(seg_img, gt_img))
        # dices.append(dice(seg_img, gt_img))
        # recalls.append(recall(seg_img, gt_img))
        # SENs.append(sen(seg_img, gt_img))
        # SPEs.append(spe(seg_img, gt_img))


    # values = np.asarray(values)
    hausdos = np.asanyarray(hausdos)
    voes = np.asanyarray(voes)
    dices = np.asanyarray(dices)
    recalls = np.asanyarray(recalls)
    SENs = np.asanyarray(SENs)
    SPEs = np.asanyarray(SPEs)




    print(f"HAUSDOSs: {hausdos.mean()}")
    print(f"VOEs: {voes.mean()}")
    print(f"RECALLs: {recalls.mean()}")
    print(f"dices: {dices.mean()}")
    print(f"SENs: {SENs.mean()}")
    print(f"SPEs: {SPEs.mean()}")


    return hausdos.mean(), voes.mean(), recalls.mean(), dices.mean(),SENs.mean(),SPEs.mean()










def dice_WT(seg,ground):
    #computs false negative rate
    # TP = np.sum(np.multiply(ground, seg))

    FN = 0
    FP = 0
    TN = 0
    TP = 0
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if seg[i][j]==0 and ground[i][j]==1:
                FN = FN+1
            if seg[i][j]==1 and ground[i][j]==0:
                FP = FP+1
            if seg[i][j] == 0 and ground[i][j] == 0:
                 TN = TN + 1
            if seg[i][j] == 1 and ground[i][j] == 1:
                TP = TP + 1

    # PPV
    if (TP+FP)==0:
        PPV = 0
    else:
        PPV = TP / (TP + FP)
    # 敏感性
    if (TP+FN)==0:
        SE = 0
    else:
        SE = TP / (TP + FN)
    # 特异性
    if (TN+FP)==0:
        SP = 0
    else:
        SP = (TN) / (TN + FP)
    # DICE
    if (FP+2*TP+FN)==0:
        dice = 0
    else:
        dice = (2*TP)/(FP+2*TP+FN)
    return PPV, SE, SP, dice, FN, FP, TN, TP

def sensitivity_of_brats_data_set(gt_path, seg_path, type_idx):
    gt_names = os.listdir(gt_path)
    seg_names = os.listdir(seg_path)
    assert (len(gt_names) == len(seg_names))
    values = []
    PPVs=[]
    SEs=[]
    SPs=[]
    dices=[]
    FNs=[]
    FPs=[]
    TNs=[]
    TPs=[]
    for name in seg_names:

        seg_img = cv2.imread(f"{seg_path}/{name}",cv2.IMREAD_UNCHANGED)[:,:,0]
        seg_img[seg_img < 255] = 1
        seg_img[seg_img == 255] = 0

        #seg_img[seg_img == 38] = 1
        #seg_img[seg_img == 125] = 1
        #seg_img[seg_img == 34] = 1
        name = name.split(".")[0]
        gt_img = cv2.imread(f"{gt_path}/{name}_seg.png", cv2.IMREAD_UNCHANGED)
        gt_img[gt_img>0] = 1
        PPV, SE, SP, dice, FN, FP, TN, TP = dice_WT(np.asarray(seg_img), np.asarray(gt_img))
        PPVs.append(PPV)
        SEs.append(SE)
        SPs.append(SP)
        dices.append(dice)
        FNs.append(FN)
        FPs.append(FP)
        TNs.append(TN)
        TPs.append(TP)

    # values = np.asarray(values)
    PPVs = np.asanyarray(PPVs)
    SEs = np.asanyarray(SEs)
    SPs = np.asanyarray(SPs)
    dices = np.asanyarray(dices)
    FNs = np.asanyarray(FNs)
    FPs = np.asanyarray(FPs)
    TNs = np.asanyarray(TNs)
    TPs = np.asanyarray(TPs)




    print(f"PPVs: {PPVs.mean()}")
    print(f"SEs: {SEs.mean()}")
    print(f"SPs: {SPs.mean()}")
    print(f"dices: {dices.mean()}")
    print(f"FNs: {FNs.mean()}")
    print(f"FPs: {FPs.mean()}")
    print(f"TNs: {TNs.mean()}")
    print(f"TPs: {TPs.mean()}")

    print(f"PPVs: {PPVs.sum() / 7400}")
    print(f"SEs: {SEs.sum() / 7400}")
    print(f"SPs: {SPs.sum() / 7400}")
    print(f"dices: {dices.sum() / 7400}")
    print(f"FNs: {FNs.sum() / 7400}")
    print(f"FPs: {FPs.sum() / 7400}")
    print(f"TNs: {TNs.sum() / 7400}")
    print(f"TPs: {TPs.sum() / 7400}")
    return PPVs.mean(), SEs.mean(), SPs.mean(), dices.mean(),FNs.mean(),FPs.mean(), TNs.mean(), TPs.mean()



if __name__ == '__main__':
    s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\swinResult\model\swin\pretrained_upper\iter_360k'
    #s_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\vanAllTestResult\finaltest\fcn_unet_individual_P\iter_300k'
    g_folder = r'D:\Program\PyCharm\mmsegment\githubKaggle\mmsegmentation\data\bratsIndividual4C\annotations\test'
    print(getResult(g_folder, s_folder))

    # patient_names_file = '/Users/qiranjia19961112/Desktop/NYU_RESEARCH/Experiment/Output/Bi-Trans6000/valid.txt'
