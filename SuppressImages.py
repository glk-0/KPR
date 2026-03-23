import pandas as pd
import os
import skimage
import numpy as np

part_groups = {'head': {0, 1, 2, 3, 4},
               'torso': {5, 6, 11, 12},
               'arms': {5, 6, 7, 8, 9, 10},
               'left_arm': {5, 7, 9},
               'right_arm': {6, 8, 10},
               'legs': {11, 12, 13, 14, 15, 16},
               'left_leg': {11, 13, 15},
               'right_leg': {12, 14, 16},
               'top': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
              'non_torso': {0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16},
              'all': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}

def get_ex_in_intervals(keypoint_list, bbox_interval):
    ex_interval = bbox_interval
    
    ex_idxs = [None, None]

    N = len(keypoint_list)

    in_intervals = dict()
    
    for i in range(N):
        if keypoint_list[i][2]:
            ex_idxs[0] = i
            break
    for i in range(N):
        if keypoint_list[-i-1][2]:
            ex_idxs[1] = N - i - 1
            break
    
    if ex_idxs[0] is not None:
        if ex_idxs[0] > 0:
            ex_interval[0] = (keypoint_list[ex_idxs[0]][1] + keypoint_list[ex_idxs[0]-1][1]) / 2
    else:
        ex_idxs[0] = 0

    if ex_idxs[1] is not None:
        if ex_idxs[1] < N - 1:
            ex_interval[1] = (keypoint_list[ex_idxs[1]][1] + keypoint_list[ex_idxs[1]+1][1]) / 2
    else:
        ex_idxs[1] = N-1

    for i in range(ex_idxs[0]+1, ex_idxs[1]):
        if not keypoint_list[i][2]:
            in_intervals[keypoint_list[i][0]] = [(keypoint_list[i][1] + keypoint_list[i-1][1]) / 2, (keypoint_list[i][1] + keypoint_list[i+1][1]) / 2]

    return ex_interval, in_intervals

def get_ex_in_bboxes(keypoint_list, bbox_sample, part_set):
    keypoint_x = []
    keypoint_y = []
    
    for i in range(len(keypoint_list)):
        occluded = i in part_set
        keypoint_x.append((i, keypoint_list[i][0], occluded))
        keypoint_y.append((i, keypoint_list[i][1], occluded))
    
    keypoint_x.sort(key=lambda x: x[1])
    keypoint_y.sort(key=lambda y: y[1])

    ex_interval_x, in_intervals_x = get_ex_in_intervals(keypoint_x, [bbox_sample[1], bbox_sample[3]])
    ex_interval_y, in_intervals_y = get_ex_in_intervals(keypoint_y, [bbox_sample[0], bbox_sample[2]])
    
    in_bboxes = []
    in_keypoints = set(in_intervals_x.keys()).intersection(set(in_intervals_y.keys()))

    for key in in_keypoints:
        in_bboxes.append([in_intervals_y[key][0], in_intervals_x[key][0], in_intervals_y[key][1], in_intervals_x[key][1]])

    return [ex_interval_y[0], ex_interval_x[0], ex_interval_y[1], ex_interval_x[1]], in_bboxes

def coord_in_bbox(coord, bbox):
    return coord[0] >= bbox[0] and coord[0] <= bbox[2] and coord[1] >= bbox[1] and coord[1] <= bbox[3]

def excluded_coord(coord, ex_bbox, in_bboxes):
    if coord_in_bbox(coord, ex_bbox):
        for bbox in in_bboxes:
            if coord_in_bbox(coord, bbox):
                return False
        return True
    return False

def occlude_image(annot_path, image_path, part_set):
    sample_annot = pd.read_json(annot_path)
    sample_image = skimage.io.imread(image_path)

    mpv = np.round(3 * np.sum(sample_image, axis=(0,1)) / sample_image.size)

    ex_bbox, in_bboxes = get_ex_in_bboxes(sample_annot['keypoints'][0], [0, 0, sample_image.shape[0], sample_image.shape[1]], part_set)
    
    coord_tensor = np.stack((np.tile(np.arange(sample_image.shape[0]), np.array((sample_image.shape[1], 1))).T, np.tile(np.arange(sample_image.shape[1]), np.array((sample_image.shape[0], 1)))), axis=2)

    sample_image[np.apply_along_axis(lambda x: excluded_coord(x, ex_bbox, in_bboxes), 2, coord_tensor)] = mpv

    return sample_image

def occlude_image_dir(annot_dir, image_dir, part_set, out_dir):
    path_list = os.listdir(annot_dir)

    for path in path_list:
        processed_image = occlude_image(os.path.join(annot_dir, path), os.path.join(image_dir, path[:-15]), part_set)
        skimage.io.imsave(os.path.join(out_dir, path[:-15]), processed_image)

def occlude_image_by_part(annot_dir, image_dir, part_sets, out_parent_dir):
    for part_key in part_sets.keys():
        occlude_image_dir(annot_dir, image_dir, part_sets[part_key], os.path.join(out_parent_dir, part_key + "_suppressed"))