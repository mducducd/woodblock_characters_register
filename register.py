from utils import *
from normalization import *
from evaluate import *
from libs import *
import time
import copy
import os


def shift_character_in_canvas(bin_print_image, bin_depth_image):
        # find maximum character specifications
        h_print, w_print = bin_print_image.shape
        h_rotated, w_rotated = bin_depth_image.shape
        max_size = max(h_print, w_print,h_rotated, w_rotated) 
        canvas_size = (max_size, max_size)

        #create a canvas for depth, print image with max_size
        canvas_depth_image = np.zeros(canvas_size, np.uint8)
        canvas_print_image = np.zeros(canvas_size, np.uint8)

        #move character to to center of canvas
        x_center = max_size //2
        y_center = max_size //2

        ##print image
        from_x_print = x_center - w_print //2
        to_x_print   = x_center + w_print - (w_print //2)

        from_y_print = y_center - (h_print //2)
        to_y_print = y_center + h_print - (h_print //2)
        canvas_print_image[from_x_print : to_x_print,from_y_print:to_y_print] = bin_print_image #draw image on a canvas
        
        ##depth image
        from_x_depth = x_center - w_rotated //2
        to_x_depth   = x_center + w_rotated - (w_rotated //2)

        from_y_depth = y_center - (h_rotated //2)
        to_y_depth = y_center + h_rotated - (h_rotated //2)
        canvas_depth_image[from_x_depth : to_x_depth,from_y_depth:to_y_depth] = bin_depth_image

        #add border to image  
        #plt.imshow(canvas_print_image)
        #plt.show()
        #find the best fit for for two two canvas
        prevIOU = 0
        step = 1
        movements = {
                "left": (-step,0),
                "right":(step,0),
                "top": (0, -step),
                "bottom":(0, step)
                        }
        changeX = 0 # recording changing in the x direction
        changeY = 0 # recording changing in the y direction
        stillChanging = True
        best_fit_canvas_depth = None
        best_fit_canvas_print = None
        while stillChanging:
                stillChanging = False
                for key,mov in movements.items():
                        deltaX, deltaY = movements[key]
                        while True:
                                #padding image for canvas
                                top, bottom, left, right = make_border(changeX + deltaX , changeY +  deltaY , 'depth')
                                #top_delta, bottom_delta, left_delta, right_delta = make_border(delta_width, delta_height , 'depth')
                                
                                new_canvas_depth_image = cv2.copyMakeBorder(canvas_depth_image, top , bottom , left , right , cv2.BORDER_CONSTANT,value = 0)
                                
                                top, bottom, left, right = make_border(changeX + deltaX , changeY +  deltaY  , 'print')
                                #top_delta, bottom_delta, left_delta, right_delta = make_border(delta_width, delta_height , 'print')
                                new_canvas_print_image = cv2.copyMakeBorder(canvas_print_image, top , bottom , left , right , cv2.BORDER_CONSTANT,value = 0)
                                
                                #compare current IOU
                                currentIOU = get_iou_metric(new_canvas_depth_image, new_canvas_print_image)
                                #print(currentIOU - prev)
                                if currentIOU - prevIOU <= 0:
                                        break
                                else:
                                        #update 
                                        prevIOU = currentIOU
                                        changeX += deltaX
                                        changeY += deltaY
                                        best_fit_canvas_depth = new_canvas_depth_image
                                        best_fit_canvas_print = new_canvas_print_image
                                        #plt.imshow(best_fit_canvas_depth+best_fit_canvas_print)
                                        #plt.show()
                                        stillChanging = True
        # Again iou not ssim
        sim = get_iou_metric(best_fit_canvas_depth, best_fit_canvas_print)
        return  best_fit_canvas_depth,best_fit_canvas_print, changeX, changeY, sim  

def find_best_angle(bin_depth_img, min_angle, max_angle, num_step, bin_print_img):
    """
    This function will be find best angle in range np.linspace(min_angle, max_angle, num_step)
    This function supports match_rotation function
    :param bin_depth_img: cv2 binary image
    :param min_angle:
    :param max_angle:
    :param num_step:
    :param bin_print_img: cv2 binary image
    :return: rotated angle, similarity score, best rotated image, best print image
    """
    maxNonZero = 0
    maxiou = 0
    best_fit_changeX = 0
    best_fit_changeY = 0
    canvas_depth_best_fit_image = 0
    canvas_print_best_fit_image = 0
    resized_print_img = copy.copy(bin_print_img)
    if min_angle == max_angle:
        angle_range = [min_angle]
    else:
        angle_range = np.linspace(min_angle, max_angle, num=num_step)

    for angle in angle_range:
        rotated_img = rotate_bound(copy.deepcopy(bin_depth_img), angle, bin_depth_img.shape[1] / 2,
                                   bin_depth_img.shape[0] / 2)

        
        rotated_img = normalize_mask(rotated_img)
        if rotated_img.shape[0] < resized_print_img.shape[0]:
            resized_print_img = cv2.resize(resized_print_img, rotated_img.shape[:2])
        else:
            rotated_img = cv2.resize(rotated_img, resized_print_img.shape[:2])
        # countNonZero = c(rotated_img, resized_print_img)

        # if countNonZero > maxNonZero:
        #     maxNonZero = countNonZero
        #     rotate_angle = angle
        #     best_depth_rotated_img = rotated_img
        #     best_print_img = resized_print_img
        canvas_depth_new_image,canvas_print_image,changeX, changeY, iou = shift_character_in_canvas(resized_print_img, rotated_img)
        if iou > maxiou:
            rotate_angle = angle
            maxiou = iou
            best_fit_changeX = changeX
            best_fit_changeY = changeY
            canvas_depth_best_fit_image  = canvas_depth_new_image
            canvas_print_best_fit_image = canvas_print_image
    
    return rotate_angle, maxiou, canvas_depth_best_fit_image, canvas_print_best_fit_image


def match_rotation(depth_image, print_image, normalize_depth_image_type):
    """
    Find best rotated angle
    :param depth_image: cv2 color depth image
    :param print_image: cv2 color print image
    :return: best angle, similarity score
    """
    
    if normalize_depth_image_type == 'v2':
      bin_depth_img, _, _, _ = normalize_depth_image_v2(depth_image, print_image)
    elif normalize_depth_image_type == 'v3':
      bin_depth_img, _, _ = normalize_depth_image_v3(depth_image, print_image)
    else:
      bin_depth_img, _, _ = normalize_depth_image_v4(depth_image, print_image)
    bin_print_img = normalize_print_image(print_image)
    maxNonZero = 0
    best_depth_rotated_img = None
    best_print_img = None
    min_angle = 0
    max_angle = 359
    max_depth = 2
    num_step = 60

    for depth in range(max_depth):
        rotate_angle, sim_score, best_depth_rotated_img, best_print_img = find_best_angle(bin_depth_img, min_angle,
                                                                                            max_angle, num_step,
                                                                                            copy.copy(bin_print_img))
        min_angle = rotate_angle - 3 * ((max_angle - min_angle) / num_step)
        max_angle = rotate_angle + 3 * ((max_angle - min_angle) / num_step)
    g_best_depth_rotated_img = 255 - convert_color_img(255 - best_depth_rotated_img, 'g')
    r_best_print_img = 255 - convert_color_img(best_print_img, 'r')
    stacked_img = g_best_depth_rotated_img + r_best_print_img
   
    return dict(rotate_angle=rotate_angle, sim_score=sim_score, best_depth_rotated_img=best_depth_rotated_img, best_print_img=best_print_img, stacked_img=stacked_img, bin_depth_img=bin_depth_img)

def register_v1(id):
    """
    Register 2 images (actually get different angle between direction of depth character and print character
    :param depth_image_path: path to depth image
    :param print_image_path: path to print image
    :return:
    """
   
    depth_image_path = '/content/drive/MyDrive/DATA_WORK/depth_map_img' + '/' + id + '_r.png'
    # print_image_path = train_path + id + '/' + id + '01.png'
    # print_image_path = '/content/drive/MyDrive/DATA_WORK/woodblock_labels/processed/25900_pad' '/' + id + '.png'
    print_image_path = '/content/drive/MyDrive/21072021_mocban_sokhop/full_2d_img/' + id + '.png'
    depth_image = cv2.imread(depth_image_path)
    depth_image = cv2.bitwise_not(depth_image)
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    black_pixels = np.where(
        (depth_image[:, :, 0] == 0) & 
        (depth_image[:, :, 1] == 0) & 
        (depth_image[:, :, 2] == 0)
    )

    # set those pixels to white
    depth_image[black_pixels] = [255, 255, 255]
    # im_pil = change_contrast(Image.fromarray(depth_image), 60)
    # depth_image = np.asarray(im_pil)
    print_image = cv2.imread(print_image_path)
    print_image = cv2.flip(print_image, 1)
    result1 = match_rotation(depth_image, print_image, 'v2')
    result2 = match_rotation(depth_image, print_image, 'v4')
   
    print(result2['sim_score'])
    result = result1 if result1['sim_score'] > result2['sim_score'] else result2
    result = result1
    
    concated_img = concatenate_image(depth_image, print_image, result['rotate_angle'], (256, 256))

    img_name = os.path.basename(print_image_path).replace('.png', '')
    iou = get_iou_metric(result['best_depth_rotated_img'], result['best_print_img'])
    ssim_score = ssim(result['best_depth_rotated_img'], result['best_print_img'])
    dice = get_dice_metric(result['best_depth_rotated_img'], result['best_print_img'])
    save_path = '/content/drive/MyDrive/DATA_WORK/GAN_pix2pix/register_outputs/6/'
    cv2.imwrite(save_path + img_name + '_registed.jpg', concated_img)
    cv2.imwrite(save_path + img_name.replace('.png', '') + '_stacked.jpg', result['stacked_img'])
    cv2.imwrite('/content/drive/MyDrive/DATA_WORK/GAN_pix2pix/register_outputs/good_thresh_depth/'+img_name+'.png', result['bin_depth_img'])
    with open(save_path+'register_log_4k.txt', 'a') as f:
    # # with open('/content/drive/MyDrive/DATA_WORK/woodblock_labels/register_log.txt', 'a') as f:
        f.write(img_name + ' ' +str(result['rotate_angle'])+' '+str(round(ssim_score, 5))+' '+str(round(iou, 5))+' '+str(round(dice, 5))+'\n')
    print('SSIM: ', ssim_score)
    return concated_img, result['best_depth_rotated_img'], result['best_print_img'], result['stacked_img'], result['rotate_angle'], ssim_score


if __name__ == "__main__":
    start = time.time()
    concated_img, best_depth_rotated_img, best_print_img, stacked_img, rotate_angle, ssim_score = register_v1('291412050102')
    print(ssim_score, rotate_angle)
    print(time.time()-start)
