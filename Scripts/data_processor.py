import os, cv2, numpy as np, tensorflow as tf

path_base = '/media/lewisluk/新加卷1/dataset/色彩浓艳度分类'

def normalise_imgs_in_dir(dir_name:str, target_max_length:int=512)->None:
    '''

    :param dir_name:            dir name of images to be normalised
    :param target_max_length:   images will be resize to target_max_length*target_max_length
    :return:                    None
    '''
    tf.enable_eager_execution()
    imgs = os.listdir(os.path.join(path_base, dir_name))
    for i in imgs:
        img = cv2.imread(os.path.join(path_base, dir_name, i)).astype(np.float32)
        result_img = tf.image.resize_image_with_pad(img, target_max_length, target_max_length)
        result_img = np.array(result_img, dtype=np.uint8)
        cv2.imwrite(os.path.join(path_base, dir_name, i), result_img)
        print(i)
    tf.disable_eager_execution()

def rename()->None:
    # rename filenames to number indices
    # after this images from two dirs have the same names
    imgs_input = os.listdir(os.path.join(path_base, 'A'))
    imgs_label = os.listdir(os.path.join(path_base, 'B'))
    for i in range(len(imgs_input)):
        os.rename(os.path.join(path_base, 'A', imgs_input[i]),
                  os.path.join(path_base, 'A', str(i + 1) + '.jpg'))
        os.rename(os.path.join(path_base, 'B', imgs_label[i]),
                  os.path.join(path_base, 'B', str(i + 1) + '.jpg'))
normalise_imgs_in_dir('A')
normalise_imgs_in_dir('B')

rename()