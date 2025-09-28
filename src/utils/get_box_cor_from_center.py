def get_box_cor_from_center(center):
    x, y = center
    return ((int(x - 8), int(y - 8)), (int(x + 8), int(y + 8)))