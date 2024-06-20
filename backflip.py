import os
import cv2
import torch
import numpy as np

from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def get_segment_bounding_box(mask):
    """
    Calculates the bounding box of a given mask.

    Parameters:
    mask (numpy.ndarray): The input mask.

    Returns:
    dict: A dictionary containing the coordinates of the bounding box.
          The dictionary has the following keys:
          - 'min_x': The minimum x-coordinate of the bounding box.
          - 'max_x': The maximum x-coordinate of the bounding box.
          - 'min_y': The minimum y-coordinate of the bounding box.
          - 'max_y': The maximum y-coordinate of the bounding box.
          - 'center_x': The x-coordinate of the center of the bounding box.
          - 'center_y': The y-coordinate of the center of the bounding box.
    """
    # Find the indices of non-zero elements in the mask
    indices = np.where(mask != 0)

    # Get the minimum and maximum indices along each axis (coordinates of the bounding box)
    min_x = np.min(indices[1])
    max_x = np.max(indices[1])
    min_y = np.min(indices[0])
    max_y = np.max(indices[0])

    # Calculate the center of the bounding box
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'center_x': center_x,
        'center_y': center_y
    }

def insert_element(background, element, mask, position):
    """
    Inserts an element onto a background image at a specified position.

    Parameters:
    background (numpy.ndarray): The background image.
    element (numpy.ndarray): The element to be inserted onto the background image.
    mask (numpy.ndarray): The mask indicating the region of the element to be inserted.
    position (tuple): The position (x, y) where the element should be inserted (center).

    Returns:
    numpy.ndarray: The modified background image with the element inserted.

    """
    adj_position = (position[0] - element.shape[0] // 2, position[1] - element.shape[1] // 2)

    #Element is out of bounds y axis
    if adj_position[0] + element.shape[0] > background.shape[0]:
        element = element[:background.shape[0] - adj_position[0]]
        mask = mask[:background.shape[0] - adj_position[0]]
    if adj_position[0] < 0:
        mask = mask[-adj_position[0]:]
        element = element[-adj_position[0]:]
        adj_position = (0, adj_position[1])
    # Checking if the element is out of bounds x axis
    if adj_position[1] + element.shape[1] > background.shape[1]:
        element = element[:, :background.shape[1] - adj_position[1]]
        mask = mask[:, :background.shape[1] - adj_position[1]]
    if adj_position[1] < 0:
        mask = mask[:, -adj_position[1]:]
        element = element[:, -adj_position[1]:]
        adj_position = (adj_position[0], 0)

    globalMask = np.zeros_like(background)
    globalMask[adj_position[0]:adj_position[0]+element.shape[0], adj_position[1]:adj_position[1]+element.shape[1]] = mask
    globalMask = globalMask.any(axis=2).astype(bool)

    globalImage = np.zeros_like(background)
    globalImage[adj_position[0]:adj_position[0]+element.shape[0], adj_position[1]:adj_position[1]+element.shape[1]] = element

    background[globalMask] = globalImage[globalMask]

    return background

def lama_inpaint(img: np.ndarray, img_name: str, mask_name: str, inpainted_dir: str,
                 dilated_mask_dir: str = 'dilated_masks', inpainted_imgs_dir: str = 'inpainted_imgs') -> np.ndarray:
    """
    Inpaints the given image using the corresponding inpainted and dilated mask images.

    Args:
        img (np.ndarray): The original image to be inpainted.
        img_name (str): The name of the original image file.
        mask_name (str): The name of the mask file.
        inpainted_dir (str): The directory where the inpainted images are stored.
        dilated_mask_dir (str, optional): The directory where the dilated mask images are stored. Defaults to 'dilated_masks'.
        inpainted_imgs_dir (str, optional): The directory where the inpainted images are stored. Defaults to 'inpainted_imgs'.

    Returns:
        np.ndarray: The inpainted image.
    """
    img_name_no_ext = img_name[:img_name.find('.')]
    dilated_mask_file = f'{img_name_no_ext}_mask00{int(mask_name[:mask_name.find('.')]) + 1}.png'

    dilated_mask = cv2.imread(os.path.join(inpainted_dir, dilated_mask_dir, dilated_mask_file))
    inpainted = cv2.imread(os.path.join(inpainted_dir, inpainted_imgs_dir, dilated_mask_file))
    
    boolMask = np.all(dilated_mask == 255, axis=-1)
    img[boolMask] = inpainted[boolMask]

    return img

def backflip(img: np.ndarray, img_name: str, possible_aug: list, aug_prob: list, num_of_segments: int, segment_dir: str = 'segments',
             inpaint_method: str = 'telea', lama_inpainted_dir: str = None) -> np.ndarray:
    """
    Apply backflip augmentation to an image.

    Args:
        img (np.ndarray): The input image.
        img_name (str): The name of the image.
        possible_aug (list): List of possible augmentation functions.
        aug_prob (list): List of probabilities for each augmentation function.
        num_of_segments (int): Number of segments to select for augmentation.
        segment_dir (str, optional): Directory where the segments are stored. Defaults to 'segments'.
        inpaint_method (str, optional): Inpainting method to fill the background. Defaults to 'telea'.

    Returns:
        np.ndarray: The augmented image.
    """
    
    # Check inpaint method
    if inpaint_method not in ['telea', 'ns', 'mean', 'median', 'lama']:
        raise ValueError('Inpaint method not recognized.')

    # Get the segment names
    folder_name = img_name[:img_name.rfind('.')]
    segment_names = [f for f in os.listdir(os.path.join(segment_dir, folder_name)) if os.path.isfile(os.path.join(segment_dir, folder_name, f))]

    # Selected segments
    if num_of_segments > len(segment_names):
        selected_segments = segment_names
    else:
        selected_segments = np.random.choice(segment_names, num_of_segments, replace=False)

    unchanged_img = img.copy()

    # Augmentation loop
    for segment in selected_segments:
        # Opening mask
        mask = cv2.imread(os.path.join(segment_dir, folder_name, segment))
        mask_bool = mask.any(axis=2).astype(bool)

        # Filling background
        if inpaint_method == 'telea':
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            inpaint = cv2.inpaint(img, mask_gray, 3, cv2.INPAINT_TELEA)
            img[mask_bool] = inpaint[mask_bool]
        elif inpaint_method == 'ns':
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            inpaint = cv2.inpaint(img, mask_gray, 3, cv2.INPAINT_NS)
            img[mask_bool] = inpaint[mask_bool]
        elif inpaint_method == 'mean':   
            edges = cv2.subtract(cv2.dilate(mask, np.ones((6, 6), np.uint8), iterations=2), mask)
            edges = cv2.bitwise_and(img, edges)
            median_color = np.mean(edges[edges.sum(axis=2) > 0], axis=0)
            img[mask_bool] = median_color
        elif inpaint_method == 'median':
            mask_bool = mask.any(axis=2).astype(bool)
            edges = cv2.subtract(cv2.dilate(mask, np.ones((6, 6), np.uint8), iterations=2), mask)
            edges = cv2.bitwise_and(img, edges)
            median_color = np.median(edges[edges.sum(axis=2) > 0], axis=0)
            img[mask_bool] = median_color
        elif inpaint_method == 'lama':
            if lama_inpainted_dir is None:
                raise ValueError('Lama inpainted directory not provided.')
            img = lama_inpaint(img, img_name, segment, lama_inpainted_dir)

        # # Augmenting the segment
        bbox = get_segment_bounding_box(mask)
        cut_img = unchanged_img[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x']]
        cut_mask = mask[bbox['min_y']:bbox['max_y'], bbox['min_x']:bbox['max_x']]

        selected_augmentation = np.random.choice(possible_aug, p=aug_prob)
        augmented_segment = selected_augmentation(cut_img, cut_mask)

        # Inserting the augmented segment
        img = insert_element(img, augmented_segment['img'], augmented_segment['mask'], (bbox['center_y'], bbox['center_x']))

        return img
    
def resize_image(image, length):
    """
    Resize the given image while maintaining its aspect ratio.

    Parameters:
    image (numpy.ndarray): The input image to be resized.
    length (int): The desired length (either width or height) of the resized image.

    Returns:
    numpy.ndarray: The resized image.

    """
    # Get the current dimensions of the image
    height, width = image.shape[:2]
    
    if height > width:
        is_height = True
    else:
        is_height = False

    if is_height:
        # Calculate the new width based on the provided height
        new_width = int((length / height) * width)
        # Resize the image using the new dimensions
        resized_image = cv2.resize(image, (new_width, length))
    else:
        # Calculate the new height based on the provided width
        new_height = int((length / width) * height)
        # Resize the image using the new dimensions
        resized_image = cv2.resize(image, (length, new_height))
    
    return resized_image

def pre_segmentate(image_dir: str, size: int, max_num_of_segments: int = 5, output_dir_masks: str = 'segments', output_dir_img: str = None,
                   model_checkpoint: str = 'models/defaultModel.pth'):
    """
    Pre-segmentates images in the given directory.

    Args:
        image_dir (str): The directory path containing the images to be pre-segmented.
        size (int): The desired size of the pre-segmented images.
        max_num_of_segments (int, optional): The maximum number of segments to be generated for each image. Defaults to 5.
        output_dir_masks (str, optional): The directory path to save the generated segment masks. Defaults to 'segments'.
        output_dir_img (str, optional): The directory path to save the resized images. Defaults to None.

    Returns:
        None
    """
    # Create output directories if they don't exist
    if not os.path.isdir(output_dir_masks):
        os.mkdir(output_dir_masks)
    if output_dir_img is not None and not os.path.isdir(output_dir_img):
        os.mkdir(output_dir_img)
    
    # Loading sam model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using:',device)
    sam = sam_model_registry['default'](checkpoint=model_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Iterate over all images in the directory
    for file in tqdm(os.listdir(image_dir)):
        image_name = file[:file.rfind('.')]

        # Check if image was already segmented
        if os.path.isdir(os.path.join(output_dir_masks, image_name)):
            continue         
        
        # Load image
        image = cv2.imread(os.path.join(image_dir, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image(image, size)

        # Save resized image
        if output_dir_img is not None:
            cv2.imwrite(os.path.join(output_dir_img, image_name + '.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Generate masks
        masks = mask_generator.generate(image)

        # Segment selection #################################################################
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)

        if len(masks) < max_num_of_segments:
            selected = masks
        else:
            selected = masks[:max_num_of_segments]

        # Save masks
        os.mkdir(os.path.join(output_dir_masks, image_name))
        for i,mask in enumerate(selected):
            toSave = mask['segmentation'].astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir_masks, image_name, f'{i}.png'), toSave)