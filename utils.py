import numpy as np
import scipy.misc as spm
from keras.preprocessing.image import *



def normalize(images, new_max, new_min, old_max=None, old_min=None):
    if old_min is None:
        old_min = np.min(images)
    if old_max is None:
        old_max = np.max(images)

    return (images - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min


def crop_image(img, crop_window):
    return img[crop_window[0][0]:img.shape[0] - crop_window[0][1], 
               crop_window[1][0]:img.shape[1] - crop_window[1][1],
              :]
    ## Explanation
    #     2       3
    #  ___|_______|____
    #  |              |
    #0_|              |
    #  |              |
    #  |              |
    #1_|              |
    #  |______________|
    
    

def get_cropped_shape(img_shape, crop_window):
    return (img_shape(0) - crop_window[0][0] - crop_window[0][1],
            img_shape[1] - crop_window[1][0] - crop_window[1][1],
            img_shape[2])
    '''
    return (input_shape[0],
            input_shape[1] - self.cropping[0][0] - self.cropping[0][1],
            input_shape[2] - self.cropping[1][0] - self.cropping[1][1],
            input_shape[3])
    '''
def resize_image(img, size):
    return spm.imresize(img, size)


def extract_filename(path):
    return path.split('/')[-1]


def adjust_path(path, new_location):
    return '%s/%s' % (new_location, extract_filename(path))


def load_MultipleImages(paths, img_height, img_width, grayscale=True):    
    images = []
    for i,p in enumerate(paths):        
        try:
            #print('loading: ',p)
            img = load_img(p, target_size=(img_height, img_width),grayscale=grayscale)
            if grayscale:
                img = np.reshape(img,(img_height, img_width,1))
            img = np.array(img)
            images.append(img)            
        except:
            pass        

    return np.array(images)

def load_SingleImage(path, img_height, img_width, grayscale=True):    
    try:
        img = load_img(path, target_size=(img_height, img_width),grayscale=grayscale)            
        if grayscale:
            img = np.reshape(img,(img_height, img_width,1))
        img = np.array(img)        
    except:
        pass        

    return img