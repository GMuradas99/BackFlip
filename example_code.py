import os
import cv2
import urllib.request

from backflip import backflip, pre_segmentate
from local_augmentations import ver_flip, hor_flip

image_dataset = 'datasets/Small_Lapis'
resize_length = 224
max_num_of_segments = 5

if not os.path.isdir('models/'):
    print('Downloading Model')
    os.mkdir('models/')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', 'models/defaultModel.pth')

# Pre compute the segments for all the images (only need to do it once for all the experiments)
pre_segmentate(image_dataset, resize_length, max_num_of_segments)

""" Augment one single image with backflip """
""" ###################################### """

# Load the image
image_name = 'wu-guanzhong_attachment-2001.png'
img = cv2.cvtColor(cv2.imread(os.path.join('resized_images', image_name)), cv2.COLOR_BGR2RGB)

# All the possible local augmentations
augmentations = [
    hor_flip,
    ver_flip
]
# The probabilities of applying each augmentation
probabilities = [
    0.8, 
    0.2
]
# Number of segments to augment
num_segments = 3

# Call backflip
augmented_image = backflip(img, image_name, augmentations, probabilities, num_segments)
cv2.imshow('augmented_image',augmented_image)
cv2.waitKey(0)