from utils import *
from libs import *
import copy

def normalize_mask(thresh):
    """
    Get character image and shift center of character to center of image(a part of image can be disappeared)
    :param thresh: binary image
    :return: image that contain image, and center of image is center of character
    """
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    thresh = crop_char(thresh, contours)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x_center, y_center = find_center(list_contours(contours), thresh)
    shifted_img = shift_image(thresh, thresh.shape[1] / 2 - x_center, thresh.shape[0] / 2 - y_center)
    # plt.imshow(cv2.circle(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB), (int(x_center), int(y_center)), 10, (255, 0, 0), 1), cmap='gray')
    return resize_image_to_square(shifted_img, max(shifted_img.shape[:2]))

def normalize_print_image(print_image, image_size=256):
    """
    Get character area that center of character is same as center of image
    :param print_image: the cv2 color square image that was padded
    :param image_size: tgt image size
    :return: character area
    """
    print_image = resize_image_to_square(print_image, image_size, [255, 255, 255])
    gray = cv2.cvtColor(print_image, cv2.COLOR_BGR2GRAY)
    ret, bin_print_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    character_image = normalize_mask(bin_print_image)
    
    # kernel = np.ones((3, 3), np.uint8)
    # character_image = cv2.erode(character_image, kernel, iterations=1)
    return character_image

def normalize_depth_image(depth_image, thresh=255):
    """
    Get the mask of depth image
    :param depth_image: cv2 color image
    :return: binarized image
    """
    ### Normalize depth_image mask
    depth_image = cv2.resize(depth_image, (256, 256))
    hsvImg = cv2.cvtColor(depth_image, cv2.COLOR_BGR2HSV)
    # increase contrast
    value = 90

    vValue = hsvImg[..., 2]
    hsvImg[..., 2] = np.where((255 - vValue) < value, 255, vValue + value)
    gray = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(hsvImg, cv2.COLOR_BGR2GRAY)
    thresh = 255-cv2.adaptiveThreshold(gray,thresh,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    # thresh = 255 - cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                                      cv2.THRESH_BINARY, 11, 1)
    thresh1 = normalize_mask(remove_noise(thresh))
    # kernel = np.ones((3, 3), np.uint8)
    # thresh1 = cv2.dilate(thresh1, kernel, iterations=1)

    return thresh1

def normalize_depth_image_v2(depth_image, print_image):
    """
    Find the best depth image correspond to print image
    The idea that best depth image will have least diff of dense ratio. So, this function will find threshold parameter
    and num iteration of erode/dilate to get least diff of dense ratio
    :param depth_image: cv2 color depth image
    :param print_image: cv2 color image
    :return: best depth image and best parameter
    """
    print_image = resize_image_to_square(print_image, dst=256, color=[255, 255, 255])
    
    print_image = cv2.cvtColor(print_image, cv2.COLOR_BGR2GRAY)
    ret, bin_print_image = cv2.threshold(print_image, 120, 255, cv2.THRESH_BINARY_INV)
    # bin_print_image = remove_noise(bin_print_image)
    print_dense = estimate_dense_ratio_wrap(bin_print_image)
    depth_image = cv2.resize(depth_image, (256, 256))
    # bin_depth_image = normalize_depth_image(depth_image)
    # hsvImg = cv2.cvtColor(depth_image, cv2.COLOR_BGR2HSV)
    # # increase contrast
    # value = 90

    # vValue = hsvImg[..., 2]
    # hsvImg[..., 2] = np.where((255 - vValue) < value, 255, vValue + value)
    # gray = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

    best_diff = 1
    best_score = -1

    for th in range(50, 140):
      try:
        ret, bin_depth_image = cv2.threshold(copy.deepcopy(depth_image), th, 255, cv2.THRESH_BINARY_INV)
        depth_dense = estimate_dense_ratio_wrap(bin_depth_image)
        # print(th, abs(depth_dense - print_dense))
        dense = abs(depth_dense - print_dense)
        if dense < best_diff:
            need_erode = True if depth_dense > print_dense else False
            best_diff = abs(depth_dense - print_dense)
            best_depth_image = bin_depth_image
            best_thresh = th #+ 10
            best_dense = dense
      except Exception:
        continue

    ret, best_depth_image = cv2.threshold(depth_image, best_thresh, 255, cv2.THRESH_BINARY_INV)
    num_iter = 0
    return best_depth_image, bin_print_image, best_diff, best_thresh

def normalize_depth_image_v3(depth_image, print_image):
    """
    Find the best depth image correspond to print image
    The idea that best depth image will have least diff of dense ratio. So, this function will find threshold parameter
    and num iteration of erode/dilate to get least diff of dense ratio
    :param depth_image: cv2 color depth image
    :param print_image: cv2 color image
    :return: best depth image and best parameter
    """
    gray1 = cv2.cvtColor(cv2.resize(print_image, (256,256)), cv2.COLOR_BGR2GRAY)
    gray1 = resize_image_to_square(gray1, 256, [255,255,255])
    ret, bin_print_image = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV)
    gray2 = cv2.cvtColor(cv2.resize(depth_image, (256,256)), cv2.COLOR_BGR2GRAY)

    

    best_score = -1
    for th in range(90, 150):
      best_thresh = th 
      try:
        ret, bin_depth_image = cv2.threshold(gray2, th, 255, cv2.THRESH_BINARY_INV)
        score, square2, dense_diff = sim_score(bin_print_image, bin_depth_image)
        if score > best_score and dense_diff < 0.05:
          best_score = score
#           best_square = square2
          best_thresh = th 
#           best_bin_depth_image = bin_depth_image
          # best_dense_diff = dense_diff
      except Exception:
        continue

  
    ret, best_depth_image = cv2.threshold(gray2, best_thresh, 255, cv2.THRESH_BINARY_INV)
 
    return remove_noise(best_depth_image), best_thresh, best_score

def normalize_depth_image_v4(depth_image, print_image):
    """
    Find the best depth image correspond to print image
    The idea that best depth image will have least diff of dense ratio. So, this function will find threshold parameter
    and num iteration of erode/dilate to get least diff of dense ratio
    :param depth_image: cv2 color depth image
    :param print_image: cv2 color image
    :return: best depth image and best parameter
    """

    gray1 = cv2.cvtColor(cv2.resize(print_image, (256,256)), cv2.COLOR_BGR2GRAY)
    gray1 = resize_image_to_square(gray1, 256, [255,255,255])
    ret, bin_print_image = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY_INV)
    gray2 = cv2.cvtColor(cv2.resize(depth_image, (256,256)), cv2.COLOR_BGR2GRAY)

    list_dense_diff = []
    list_ssim = []
    current_thresh = 90
    for th in range(90, 150):
      try:
        ret, bin_depth_image = cv2.threshold(gray2, th, 255, cv2.THRESH_BINARY_INV)
        score, square2, dense_diff = sim_score(bin_print_image, bin_depth_image)
        list_ssim.append(score)
        # dense_diff = abs(estimate_dense_ratio(center_crop(bin_depth_image, (max_size,max_size)))-estimate_dense_character_ratio(bin_print_image))
        list_dense_diff.append(dense_diff)
      except Exception:
        current_thresh += 1
        continue
    current_ssim = 0
    for i in range(len(list_ssim)):
      if list_ssim[i] > current_ssim:# and list_dense_diff[i] < max_accepted_diff_dense:
        current_ssim = list_ssim[i]
    best_thresh = current_thresh + list_ssim.index(current_ssim)

    ret, best_bin_depth_image = cv2.threshold(gray2, best_thresh, 255, cv2.THRESH_BINARY_INV)
    print(best_thresh)
    
    return remove_noise(best_bin_depth_image), best_thresh, best_thresh
