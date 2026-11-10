import os
import time
import torch
import random
import copy
import numpy as np


def occluded_image_ratio(img, occ, factor=1.0):
    W, H = img.size
    occ_w, occ_h = occ.size

    new_w, new_h = min(W-1, int(factor * occ_w)), min(H-1, int(factor * occ_h))
    occ = occ.resize((new_w, new_h))

    center_x = random.choice(range(0, W))
    center_y = random.choice(range(0, H))
    
    start_x = center_x - new_w // 2
    start_y = center_y - new_h // 2
    
    end_x = center_x + (new_w + 1) // 2
    end_y = center_y + (new_h + 1) // 2
    # occlude the img
    # box = (x, y, x+new_w, y+new_h)
    box = (start_x, start_y, end_x, end_y)
    img.paste(occ, box)
    
    # cal the corresponding mask
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(W-1, end_x)
    end_y = min(H-1, end_y)
    mask = np.zeros((H, W))
    mask[start_y:end_y, start_x:end_x] = 1.0

    ratio = ((end_y - start_y) * (end_x - start_x)) / float(H * W)

    return img, mask, ratio

def cal_IoU(mask1, mask2):
    inter = np.sum(mask1 * mask2)
    union = np.sum(np.clip(mask1 + mask2, 0, 1)) + 1e-10
    return inter / union

def cal_similarity_label(grids, mask):
    scores = []
    for i, grid in enumerate(grids):
        score = cal_IoU(grid, mask)
        scores.append(score)
    occ_label = np.argmax(scores)
    return occ_label

def find_smooth_start (position, axis):
    idx = axis.tolist().index(position)
    poss_labels = []
    if idx>0:
        poss_labels.append(axis[idx-1])
    return poss_labels

def find_smooth_end (position, axis):
    idx = axis.tolist().index(position)
    poss_labels = []
    if idx<len(axis)-1:
        poss_labels.append(axis[idx+1])
    return poss_labels

def find_poss_labels(cad_x_start,cad_x_end,cad_y_start,cad_y_end,start,end):
    cad_smooth = []
    dist = 1
    start, end = list(start), list(end)
    temp_start , temp_end = copy.deepcopy(start) , copy.deepcopy(end)
    if len(cad_x_start)!=0:
        temp_start[0] = cad_x_start[dist-1]
        cad_smooth.append((tuple(temp_start),tuple(end)))
        temp_start = copy.deepcopy(start)
    if len(cad_y_start)!=0:
        temp_start[1]= cad_y_start[dist-1]
        cad_smooth.append((tuple(temp_start),tuple(end)))
        temp_start =copy.deepcopy(start)
    if len(cad_x_end)!=0:
        temp_end [0] = cad_x_end[dist-1]
        cad_smooth.append((tuple(start),tuple(temp_end)))
        temp_end = copy.deepcopy(end)
    if len(cad_y_end)!=0:
        temp_end [1] = cad_y_end[dist-1]
        cad_smooth.append((tuple(start),tuple(temp_end)))
        temp_end = copy.deepcopy(end)
    return cad_smooth
    
        
def get_grids(H, W, N):
    grid_ori = np.zeros((H, W))

    x_axis = np.linspace(0, W, N+1, True, dtype=int)
    y_axis = np.linspace(0, H, N+1, True, dtype=int)

    vertex_set = []
    for y in y_axis:
        for x in x_axis:
            vertex_set.append((y, x))

    grids = [grid_ori]
    idx = 1
    grids_dic = {}
    grids_idx = {}
    for start in vertex_set:
        for end in vertex_set:
            if end[0] > start[0] and end[1] > start[1]:
                grid = grid_ori.copy()
                grid[start[0]:end[0], start[1]:end[1]] = 1.0
                grids.append(grid)
                cad_x_start = find_smooth_start(start[0],y_axis)
                cad_y_start = find_smooth_start(start[1],x_axis)
                
                cad_x_end = find_smooth_end(end[0],y_axis)
                cad_y_end = find_smooth_end(end[1],x_axis)
                
                grids_idx[(start,end)] = idx
                grids_dic[idx] = find_poss_labels(cad_x_start,cad_x_end,cad_y_start,cad_y_end, start, end)
                idx+=1
    # print('grids_idx', grids_idx)        
    smooth_labels = {}
    for i, grid in enumerate(grids[1:]):
        poss_cads = grids_dic[i+1]
        poss_labels = []
        for location in poss_cads:
            poss_labels.append(grids_idx[location])
        smooth_labels[i+1] = poss_labels
    
 
    return grids, smooth_labels
