from libs import *
import imutils

def find_center(list2, mask):
    # scipy.ndimage.measurements.center_of_mass¶
    """
    Find center of the character(mask)
    :param list2: list of contours
    :param mask: mask indicates the position of character
    :return: center of character
    """
    cnts = cv2.drawContours(mask, list2, -1, (0, 255, 0), 1)

    kpCnt = len(list2)

    x = 0
    y = 0

    for kp in list2:
        x = x + kp[0][0]
        y = y + kp[0][1]

    # cv2.circle(mask, (np.uint8(np.ceil(x/kpCnt)), np.uint8(np.ceil(y/kpCnt))), 1, (255, 255, 255), 1)
    return x / kpCnt, y / kpCnt  # x_center, y_center

# def shift_image(image, x, y):
#   M = np.float32([[1, 0, x], [0, 1, y]])
#   return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def list_contours(contours):
    list = []
    max_len = 0
    for cnt in contours:
        for p in cnt:
            list.append(p)
    return list


def filter_much_less_cnts(contours, min_area=20):
    """
    remove contours that have the area is smaller than min_area
    :param contours: list of contours (outputs of cv2.findContours)
    :param min_area: min contour area
    :return: list of contours is more than min_area
    """
    cnt_list = []  # to avoid error
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            # contours.remove(cnt)
            cnt_list.append(cnt)
    return list_contours(cnt_list)

def get_cnt_area(bin_image):
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = 0
    for cnt in contours:
        area += cv2.contourArea(cnt) 
    return area


def rotate_img(image, angle):
    """
    Rotate image angle (counter clockwise if angle > 0) at the center of image
    :param image: cv2 image
    :param angle: angle rotation
    :return: rotated image
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by x degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image but hold the ratio between 2 edges, besides, resize width/height to target width/height
    :param image: cv2 image
    :param width: None or target width
    :param height: None or height width
    :param inter: resize methods
    :return: resized image
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_image_to_square(image, dst=256, color=[0, 0, 0]):
    """
    Resize with padding, first resize image such that max edge get dst, then pad 2 sides of the other edge to dest
    :param image: cv2 image
    :param dst: target size
    :param color: constant pad value
    :return: square image
    """
    desired_size = dst

    im = image
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = color
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im

def remove_noise(bin_depth_image):
    def fill_noise(thresh1):
        h, w = thresh1.shape
        cnts = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        mask = np.ones(thresh1.shape[:2], dtype="uint8") * 255
        # loop over the contours
        for c in cnts:
            # if the contour is bad, draw it on the mask
            if cv2.contourArea(c)<(h*w/1200):
                cv2.drawContours(mask, [c], -1, 0, -1)
        # remove the contours from the image and show the resulting images
        thresh1 = cv2.bitwise_and(thresh1, thresh1, mask=mask)
        return thresh1
    return 255-fill_noise(255-fill_noise(bin_depth_image))

def rotate_bound(image, angle, x_center, y_center):
    """
    Rotate image at (x_center, y_center) and do not lose any part of image
    :param image: cv2 image
    :param angle: counter-clockwise angle
    :param x_center: x-coordinate of center
    :param y_center: y-coordinate of center
    :return: rotated image
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (x_center, y_center)
    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def pad_img(img, pad_value=20):
  # padding 20px to print_image mask
  h ,w = img.shape
  img = cv2.resize(img, (h-pad_value,w-pad_value)) ######################
  ht, wd = img.shape
  # result = np.full((hh,ww), color, dtype=np.uint8)
  thresh2 = np.zeros((h, w))
  # compute center offset
  xx = (h - wd) // 2
  yy = (w - ht) // 2
  # copy img image into center of result image
  thresh2[yy:yy+ht, xx:xx+wd] = img

  return thresh2


def make_border(valueX, valueY, name):
    top = bottom = left = right = 0
    if valueX < 0:
        left = valueX
        right = 0
    else:
        left = 0
        right = valueX
    if valueY < 0:
        top = valueY
        bottom = 0
    else:
        top = 0
        bottom = valueY
    if name == 'depth':
        return abs(top), abs(bottom), abs(left), abs(right)
    else:
        return abs(bottom), abs(top), abs(right), abs(left)

def shift_image(img, x, y):
    """
    Pad into image to center of character is center of image
    :param img: cv2 binary image
    :param x: x_center_of_image - x_center_of_character
    :param y: y_center_of_image - y_center_of_character
    :return: Padded image that center of character lies on center of image
    """
    x_abs = int(round(abs(x), 0))
    y_abs = int(round(abs(y), 0))
    if x < 0 and y < 0:
        pad_img = cv2.copyMakeBorder(img, top=y_abs, bottom=0, left=x_abs, right=0, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    elif x < 0 and y > 0:
        pad_img = cv2.copyMakeBorder(img, top=y_abs, bottom=0, left=0, right=x_abs, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    elif x > 0 and y < 0:
        pad_img = cv2.copyMakeBorder(img, top=0, bottom=y_abs, left=x_abs, right=0, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    else:
        pad_img = cv2.copyMakeBorder(img, top=0, bottom=y_abs, left=0, right=x_abs, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
    return pad_img


def crop_char(thresh, contours):
  """
  Crop image based on contours, find the most top, bot, left, right points
  :param thresh: binary image
  :param contours: list of contours
  :return: crop of original image that contains character
  """
  list1 = filter_much_less_cnts(contours)
  xmax = max(i[0][0] for i in list1)
  xmin = min([i[0][0] for i in list1])
  ymax = max(i[0][1] for i in list1)
  ymin = min([i[0][1] for i in list1])

  if (xmax - xmin) > (ymax -ymin):
    image = image_resize(thresh[ymin:ymax, xmin:xmax], width = 256)
  else:
    image = image_resize(thresh[ymin:ymax, xmin:xmax], height = 256)
  #   if (xmax-112) > (112-xmin):
  #     image = image_resize(thresh[ymin:ymax, 224-xmax:xmax], width = 224) #xmax, ymax, xmin, ymin
  #   else:
  #     image = image_resize(thresh[ymin:ymax, xmin:224-xmin], width = 224)
  # else:
  #   if (ymax-112) > (112-ymin):
  #     image = image_resize(thresh[224-ymax:ymax, xmin:xmax], height = 224)
  #   else:
  #     image = image_resize(thresh[ymin:224-ymin, xmin:xmax], height = 224)
  return image

def find_center(list2, mask):
    # scipy.ndimage.measurements.center_of_mass¶
    """
    Find center of the character(mask)
    :param list2: list of contours
    :param mask: mask indicates the position of character
    :return: center of character
    """
    cnts = cv2.drawContours(mask, list2, -1, (0, 255, 0), 1)

    kpCnt = len(list2)

    x = 0
    y = 0

    for kp in list2:
        x = x + kp[0][0]
        y = y + kp[0][1]

    # cv2.circle(mask, (np.uint8(np.ceil(x/kpCnt)), np.uint8(np.ceil(y/kpCnt))), 1, (255, 255, 255), 1)
    return x / kpCnt, y / kpCnt  # x_center, y_center

# def shift_image(image, x, y):
#   M = np.float32([[1, 0, x], [0, 1, y]])
#   return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def remove_noise(bin_depth_image):
    def fill_noise(thresh1):
        h, w = thresh1.shape
        cnts = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        mask = np.ones(thresh1.shape[:2], dtype="uint8") * 255
        # loop over the contours
        for c in cnts:
            # if the contour is bad, draw it on the mask
            if cv2.contourArea(c)<(h*w/1200):
                cv2.drawContours(mask, [c], -1, 0, -1)
        # remove the contours from the image and show the resulting images
        thresh1 = cv2.bitwise_and(thresh1, thresh1, mask=mask)
        return thresh1
    return 255-fill_noise(255-fill_noise(bin_depth_image))

def estimate_dense_ratio(bin_image):
    """
    This function is used to find the best threshold to binarize the depth image
    The dense ratio is num pixels in character / num pixels of image
    :param bin_image: binary image
    :return: ratio between num pixels in character and num pixels of image
    """
    thresh = bin_image
    
    # contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # thresh = crop_char(bin_image, contours)
    # thresh = resize_image_to_square(thresh)
    return np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1])
def estimate_dense_character_ratio(bin_image):
    """
    This function is used to find the best threshold to binarize the depth image
    The dense ratio is num pixels in character / num pixels of image
    :param bin_image: binary image
    :return: ratio between num pixels in character and num pixels of image
    """
    thresh = bin_image
    
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    thresh = crop_char(bin_image, contours)
    thresh = resize_image_to_square(thresh)
    return np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[1])

def estimate_dense_ratio_wrap(bin_image):
    """
    This function is used to find the best threshold to binarize the depth image
    The dense ratio is num pixels in character / num pixels of image
    :param bin_image: binary image
    :return: ratio between num pixels in character and num pixels of image
    """
    def crop_rect(img, rect):
      # get the parameter of the small rectangle
      center = rect[0]
      size = rect[1]
      angle = rect[2]
      center, size = tuple(map(int, center)), tuple(map(int, size))

      # get row and col num in img
      height, width = img.shape[0], img.shape[1]
      # print("width: {}, height: {}".format(width, height))

      M = cv2.getRotationMatrix2D(center, angle, 1)
      img_rot = cv2.warpAffine(img, M, (width, height))

      img_crop = cv2.getRectSubPix(img_rot, size, center)

      return img_crop
    img = bin_image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = np.array(filter_much_less_cnts(contours))

    rect = cv2.minAreaRect(cnt)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    img_crop = crop_rect(img, rect)

    return np.count_nonzero(img_crop) / (img_crop.shape[0] * img_crop.shape[1])


def rotate_bound(image, angle, x_center, y_center):
    """
    Rotate image at (x_center, y_center) and do not lose any part of image
    :param image: cv2 image
    :param angle: counter-clockwise angle
    :param x_center: x-coordinate of center
    :param y_center: y-coordinate of center
    :return: rotated image
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (x_center, y_center)
    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def pad_img(img, pad_value=20):
  # padding 20px to print_image mask
  h ,w = img.shape
  img = cv2.resize(img, (h-pad_value,w-pad_value)) ######################
  ht, wd = img.shape
  # result = np.full((hh,ww), color, dtype=np.uint8)
  thresh2 = np.zeros((h, w))
  # compute center offset
  xx = (h - wd) // 2
  yy = (w - ht) // 2
  # copy img image into center of result image
  thresh2[yy:yy+ht, xx:xx+wd] = img

  return thresh2


def make_border(valueX, valueY, name):
    top = bottom = left = right = 0
    if valueX < 0:
        left = valueX
        right = 0
    else:
        left = 0
        right = valueX
    if valueY < 0:
        top = valueY
        bottom = 0
    else:
        top = 0
        bottom = valueY
    if name == 'depth':
        return abs(top), abs(bottom), abs(left), abs(right)
    else:
        return abs(bottom), abs(top), abs(right), abs(left)


def convert_color_img(img, color):
    """
    Convert color of character of binary image [0, 255]
    :param img: cv2 binary image
    :param color: 'r'/'b'/'g': convert to red/blue/green
    :return: numpy color image
    """
    cv_rgb_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    np_rgb_color = np.array(cv_rgb_img)
    if color == 'r':
        color_index = 0
    elif color == 'g':
        color_index = 1
    else:
        color_index = 2
    np_rgb_color[np_rgb_color[:, :, color_index] == 0, color_index] = 255
    return np_rgb_color


def concatenate_image(depth_image, print_image, rotate_angle, dst=(256, 256)):
    """
    Rotate depth image rotate_angle and merge rotated depth image and print image, and resize to dst resolution
    :param depth_image: cv2 depth image
    :param print_image: cv2 print image
    :param rotate_angle: angle to rotate depth image
    :param dst: target resolution
    :return: merged image
    """
    cv_resized_depth = cv2.resize(rotate_img(depth_image, rotate_angle), dst)
    cv_resized_print = cv2.resize(print_image, dst)
    return np.concatenate((cv_resized_depth, cv_resized_print), axis=1)
def change_contrast(img, level):

    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def wrap(bin_image):
  def crop_rect(img, rect):
      # get the parameter of the small rectangle
      center = rect[0]
      size = rect[1]
      angle = rect[2]
      center, size = tuple(map(int, center)), tuple(map(int, size))

      # get row and col num in img
      height, width = img.shape[0], img.shape[1]
      # print("width: {}, height: {}".format(width, height))

      M = cv2.getRotationMatrix2D(center, angle, 1)
      img_rot = cv2.warpAffine(img, M, (width, height))

      img_crop = cv2.getRectSubPix(img_rot, size, center)
      
      return img_crop

  img = bin_image
  contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  cnt = np.array(filter_much_less_cnts(contours))

  rect = cv2.minAreaRect(cnt)

  # box = cv2.boxPoints(rect)
  # box = np.int0(box)

  img_crop = crop_rect(img, rect)
  return img_crop

def sim_score(bin_print_image, bin_depth_image):
  wrap1 = wrap(remove_noise(bin_print_image))
  wrap2 = wrap(remove_noise(bin_depth_image)) 
  square1 = wrap1
  square2 = wrap2
  score = -1
  for i in range(4):
    square2 = cv2.rotate(square2, cv2.cv2.ROTATE_90_CLOCKWISE)
    square2 = cv2.resize(square2, (square1.shape[1], square1.shape[0]))
    # square2 = resize_image_to_square(square2)
    ssim_score = ssim(square1, square2)
    # iou_score = get_iou_metric(square1, square2)
    if score < ssim_score:
      score = ssim_score
      best_square2 = square2
      
  return score, best_square2, abs(estimate_dense_ratio_wrap(square1)-estimate_dense_ratio_wrap(best_square2))
