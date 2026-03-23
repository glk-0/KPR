import pandas as pd
import os

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

def suppress_part_group_json(json_path, part_set, out_path):
    df = pd.read_json(json_path)
    for i in part_set:
        df['keypoints'][0][i][2] = 0.0
    df.to_json(out_path, orient='records')

def suppress_part_group_dir(dir_path, part_set, out_dir):
    for item in os.listdir(dir_path): 
        suppress_part_group_json(os.path.join(dir_path, item), part_set, os.path.join(out_dir, item))

def suppress_part_groups(dir_path, part_sets, out_parent_dir):
    for part_key in part_sets.keys():
        suppress_part_group_dir(dir_path, part_sets[part_key], os.path.join(out_parent_dir, part_key + "_suppressed"))