# BackFlip: The Impact of Local and Global Data Augmentations on Artistic Image Aesthetic Assessment

by Ombretta Strafforello, Gonzalo Muradas Odriozola, Fatemeh Behrad, Li-Wei Chen, Anne-Sofie Maerten, Derya Soydaner & Johan Wagemans

[[arXiv](https://arxiv.org/abs/2408.14173)] [[BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:jNOj_a8HTREJ:scholar.google.com/&output=citation&scisdr=ClE97pkpEO3561Ds110:AFWwaeYAAAAAZvPqz10XswVz7qpFvnNUC-PJ8hM&scisig=AFWwaeYAAAAAZvPqz8PTdMo4W2nc1Nm7jTdhtjA&scisf=4&ct=citation&cd=-1&hl=en)] 

## BackFlip

We introduce BackFlip, a local data augmentation technique designed specifically for artistic image aesthetic assessme IAA. The pipeline utilizes Unsupervised Segmentation and Inpainting to locally transform areas of the image.

![Pipeline (Wider) (2)](https://github.com/user-attachments/assets/f7656a82-bb2a-4346-a8ce-ebf24755e567)

Backflip offers a wide variety of hyperparameters for data augmentation ranging from number of transformed segments, types of augmentations or inpainting technique.


![TAD66k_trans-1](https://github.com/user-attachments/assets/4a02c9a8-8d41-46e3-9a58-46b18724f668)
![TAD66k_seg-1](https://github.com/user-attachments/assets/e59d3c4f-d027-46d2-9fc2-1405812e905d)

## Pre-segmenting a dataset

A dataset can be easily segmented before training to optimize the process by using the function `pre_segmentate()`

```
      pre_segmentate(image_dir = '/example_dataset', size = 224, output_dir_masks='pre_segmented_data/segments', output_dir_img='pre_segmented_data/resized_images')
```


## BackFlipping one image

Example code for applying BackFlip to one image can be found in [example_code.py](example_code.py). 

1. Load the image:
   ```
      image_name = 'example.png'
      img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
    ```

2. Indicate the augments and their probabilites:
    ```
    # Local augmentations
      augmentations = [
          hor_flip,
          ver_flip
      ]
      # The probabilities
      probabilities = [
          0.8, 
          0.2
      ]
    ```
3. Perform BackFlip
    ```
      # Number of segments to augment
      num_segments = 3
      
      # Call backflip
      augmented_image = backflip(img, image_name, augmentations, probabilities, num_segments)
    ```
   





## Citation
If you found this code helpful, please consider citing: 
```
@misc{strafforello2024backflipimpactlocalglobal,
      title={BackFlip: The Impact of Local and Global Data Augmentations on Artistic Image Aesthetic Assessment}, 
      author={Ombretta Strafforello and Gonzalo Muradas Odriozola and Fatemeh Behrad and Li-Wei Chen and Anne-Sofie Maerten and Derya Soydaner and Johan Wagemans},
      year={2024},
      eprint={2408.14173},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.14173}, 
}
```
