import cv2
import numpy as np
from utils.mse import mse
from utils.get_box_cor_from_center import get_box_cor_from_center
def evaluate_search_points(search_area_x1, search_area_y1, search_area, target_block_ycrcb, search_area_ycrcb, target_y, search_area_y, min_mse, points):
    for p in points:
        box = get_box_cor_from_center(p)
        if (
                box[0][1] < 0
                or box[0][0] < 0
                or box[1][1] > search_area.shape[0]
                or box[1][0] > search_area.shape[1]
            ):
            continue

        window_y = search_area_y[box[0][1] : box[1][1], box[0][0] : box[1][0]]
        window = search_area_ycrcb[box[0][1] : box[1][1], box[0][0] : box[1][0]]

        if window_y.shape != target_y.shape:
            continue

        error = mse(target_y, window_y)
        if error < min_mse:
            min_mse = error
            best_match_position = (
                    box[0][0] + search_area_x1,
                    box[0][1] + search_area_y1,
                )
            best_residual = target_block_ycrcb.astype(np.int16) - window.astype(
                    np.int16
                )
            p1 = p
    return min_mse,best_match_position,best_residual

def generate_search_points(step, p1):
    x1, y1 = p1
    points = [
            (x1, y1),
            (x1 - step, y1 - step),
            (x1, y1 - step),
            (x1 + step, y1 - step),
            (x1 - step, y1),
            (x1 + step, y1),
            (x1 - step, y1 + step),
            (x1, y1 + step),
            (x1 + step, y1 + step),
        ]
    
    return points

def convert_to_ycrcb(target_block, search_area):
    target_block_ycrcb = cv2.cvtColor(target_block, cv2.COLOR_BGR2YCrCb)
    search_area_ycrcb = cv2.cvtColor(search_area, cv2.COLOR_BGR2YCrCb)
    target_y = target_block_ycrcb[:, :, 0]
    search_area_y = search_area_ycrcb[:, :, 0]
    return target_block_ycrcb,search_area_ycrcb,target_y,search_area_y

def calculate_search_area(source_image, block_x, block_y, block_size, search_padding):
    search_area_x1 = max(0, block_x - search_padding)
    search_area_y1 = max(0, block_y - search_padding)
    search_area_x2 = min(source_image.shape[1], block_x + block_size + search_padding)
    search_area_y2 = min(source_image.shape[0], block_y + block_size + search_padding)
    return search_area_x1,search_area_y1,search_area_x2,search_area_y2