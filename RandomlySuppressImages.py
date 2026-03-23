import pandas as pd
import os
import skimage
import numpy as np

from SuppressImages import coord_in_bbox

def generate_bbox(shape, loc = 0.25, scale = 0.125):
    starting_size = np.clip(np.random.normal(loc, scale), 0, 1)
    aspect_ratio = np.clip(np.random.normal(1, 0.25), 0.5, 1.5) / np.clip(np.random.normal(1, 0.25), 0.5, 1.5)
    aspect_ratio = aspect_ratio * shape[0] / shape[1]
    width = np.sqrt(starting_size / aspect_ratio)
    height = width * aspect_ratio
    width = np.clip(width * shape[0], 0, shape[0])
    height = np.clip(height * shape[1], 0, shape[1])
    width = width / 2 #All operations from here on are from center to edge of box, which is half of width and height
    height = height / 2
    centre_x = np.random.uniform(width, shape[0] - width)
    centre_y = np.random.uniform(height, shape[1] - height)
    
    return [centre_x - width, centre_y - height, centre_x + width, centre_y + height]

def randomly_occlude_image(annot_path, image_path):
    sample_annot = pd.read_json(annot_path)
    sample_image = skimage.io.imread(image_path)

    mpv = np.round(3 * np.sum(sample_image, axis=(0,1)) / sample_image.size)

    ex_bbox = generate_bbox(sample_image.shape)
    
    coord_tensor = np.stack((np.tile(np.arange(sample_image.shape[0]), np.array((sample_image.shape[1], 1))).T, np.tile(np.arange(sample_image.shape[1]), np.array((sample_image.shape[0], 1)))), axis=2)

    sample_image[np.apply_along_axis(lambda x: coord_in_bbox(x, ex_bbox), 2, coord_tensor)] = mpv

    for key in range(len(sample_annot['keypoints'][0])): 
        if coord_in_bbox(sample_annot['keypoints'][0][key][0:2], ex_bbox):
            sample_annot['keypoints'][0][key][2] = 0.0

    return sample_annot, sample_image

def randomly_occlude_image_dir(annot_dir, image_dir, out_dir):
    path_list = os.listdir(annot_dir)

    for path in path_list:
        processed_annot, processed_image = randomly_occlude_image(os.path.join(annot_dir, path), os.path.join(image_dir, path[:-15]))
        processed_annot.to_json(os.path.join(out_dir, "annot", path), orient='records')
        skimage.io.imsave(os.path.join(out_dir, "image", path[:-15]), processed_image)