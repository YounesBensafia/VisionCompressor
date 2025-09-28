import cv2
import numpy as np
from utils.process_block_utils import generate_search_points, evaluate_search_points, convert_to_ycrcb, calculate_search_area
def process_block(source_image, target_image, block_x, block_y, block_size=16, search_padding=64, threshold=24):
    target_block = target_image[
        block_y : block_y + block_size, block_x : block_x + block_size
    ]

    search_area_x1, search_area_y1, search_area_x2, search_area_y2 = calculate_search_area(source_image, block_x, block_y, block_size, search_padding)

    search_area = source_image[
        search_area_y1:search_area_y2, search_area_x1:search_area_x2
    ]

    target_block_ycrcb, search_area_ycrcb, target_y, search_area_y = convert_to_ycrcb(target_block, search_area)

    min_mse = float("inf")
    best_match_position = None
    best_residual = None

    step = 32
    h, w = search_area.shape[:2]
    p1 = int(h // 2), int(w // 2)

    while step > 1:
        points = generate_search_points(step, p1)

        min_mse, best_match_position, best_residual = evaluate_search_points(search_area_x1, search_area_y1, search_area, target_block_ycrcb, search_area_ycrcb, target_y, search_area_y, min_mse, points)

        step = step / 2

    if best_match_position is not None:
        motion_vector = (
            best_match_position[0] - block_x,
            best_match_position[1] - block_y,
        )

        if min_mse < threshold:
            residual = np.zeros_like(best_residual)
        else:
            residual = best_residual

        return motion_vector, residual

    return None, None
