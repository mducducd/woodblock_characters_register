from libs import *


def get_iou_metric(bin_depth_img, bin_print_img, smooth=1e-7):
    """
    Get iou metric of 2 same size images
    iou = |bin_depth_img ^ bin_print_img| / |bin_depth_img v bin_print_img|
    :param bin_depth_img: best rotated cv2 binary depth image
    :param bin_print_img: binary print image
    :param smooth: coff in case union equals zero
    :return: iou metric
    """
    mask_depth_img = np.where(bin_depth_img == 255, 1, 0)
    mask_print_img = np.where(bin_print_img == 255, 1, 0)
    intersection = np.sum(np.logical_and(mask_depth_img, mask_print_img))
    union = np.sum(np.logical_or(mask_depth_img, mask_print_img))
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score

def get_intersection(bin_depth_img, bin_print_img):
    mask_depth_img = np.where(bin_depth_img == 255, 1, 0)
    mask_print_img = np.where(bin_print_img == 255, 1, 0)
    intersection = np.sum(np.logical_and(mask_depth_img, mask_print_img))
    return intersection


def get_dice_metric(bin_depth_img, bin_print_img, smooth=1e-7):
    """
    Get dice metric of 2 same size images
    dice = 2 * |bin_depth_img ^ bin_print_img| / |bin_depth_img| + |bin_print_img|
    :param bin_depth_img: best rotated cv2 binary depth image
    :param bin_print_img: binary print image
    :param smooth: coff in case union equals zero
    :return: dice metric
    """
    mask_depth_img = np.where(bin_depth_img == 255, 1, 0)
    mask_print_img = np.where(bin_print_img == 255, 1, 0)
    intersection = np.sum(np.logical_and(mask_depth_img, mask_print_img))
    mask_sum = np.sum(np.abs(mask_depth_img)) + np.sum(np.abs(mask_print_img))
    dice_score = 2 * (intersection + smooth)/(mask_sum + smooth)
    return dice_score