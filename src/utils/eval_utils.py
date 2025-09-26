import matplotlib.pyplot as plt
import numpy as np
from fastai.vision.all import (PILImage, get_image_files, load_learner)
from PIL import Image
from tqdm.auto import tqdm
from fastai.data.all import get_image_files



def resize_and_crop_center(im,img_size=128):
    # Purpose: Calculates the new dimensions for resizing the image while preserving its aspect ratio.
    new_shap=(np.array(im.shape)/np.min(im.shape)*img_size).astype(int)
    im=im.resize((new_shap[1],new_shap[0]))
    h,w=im.shape
    left=(w-img_size)/2
    top=(h-img_size)/2
    right = (w + img_size)/2
    bottom = (h + img_size)/2
    im = im.crop((left, top, right, bottom))
    return im

def get_metrics(test_img_dir_path,
                test_mask_dir_path,
                model_pickle_fpath,
                test_img_out_dir,
                img_size,
                save_test_preds
                ):
    learn=load_learner(model_pickle_fpath,cpu=True)
    test_images=get_image_files(test_img_dir_path)
    inter=0
    union=0
    acc=0
    for img_path in tqdm(test_images):
        mask_path = test_mask_dir_path/f'{img_path.stem}.png'
        im = Image.open(img_path)
        mask_im = Image.open(mask_path)

        im_resized = resize_and_crop_center(im, img_size=img_size)
        y_true = resize_and_crop_center(mask_im, img_size=img_size)
        y_true = np.array(y_true).astype(int)# convert mask to arry
        y_pred, *_ = learn.predict(PILImage(im_resized))
        y_pred = np.array(y_pred).astype(int)
        # Computes pixel-wise accuracy (fraction of pixels where
        acc += (y_true == y_pred).mean()
        # Counts overlapping defect pixels (intersection). for dice 
        inter += (y_pred*y_true).sum()
        # Counts all defect pixels in either mask (union, adjusted later for Jaccard).
        union += (y_pred+y_true).sum()
        if save_test_preds:
            fig, axarr = plt.subplots(1, 3)
            fig.suptitle(
                f'{img_path.stem} Image/True/Pred')
            axarr[0].imshow(im_resized, cmap='gray')
            axarr[1].imshow(y_true, cmap='gray')
            axarr[2].imshow(y_pred, cmap='gray')
            plt.savefig(test_img_out_dir/f'{img_path.stem}.jpg', dpi=70)
    
    # calculate evaluation metrics

    #Mean Dice coefficient (2 * intersection / union), or None if union=0 (no defects in any image/mask).
    dice_mean = 2. * inter/union if union > 0 else None
    # Mean Jaccard coefficient (intersection / (union - intersection)), or None if union=0.
    jacc_mean = inter/(union-inter) if union > 0 else None
    # acc_mean: Mean accuracy (average pixel-wise accuracy across images).
    acc_mean = acc/len(test_images)
    #Purpose: Returns a dictionary with the computed metrics.
    return {'dice_mean': dice_mean, 
            'jacc_mean': jacc_mean, 
            'acc_mean': acc_mean}
