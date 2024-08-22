import argparse  
import os  
from pathlib import Path  

import cv2  
import numpy as np  
from tqdm import tqdm  
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score  

def dice(y_true, y_pred):  
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)  

def general_dice(y_true, y_pred):  
    if y_true.sum() == 0:  
        if y_pred.sum() == 0:  
            return 1  
        else:  
            return 0  

    return dice(y_true, y_pred)  

def jaccard(y_true, y_pred):  
    intersection = (y_true * y_pred).sum()  
    union = y_true.sum() + y_pred.sum() - intersection  
    return (intersection + 1e-15) / (union + 1e-15)  

def general_jaccard(y_true, y_pred):  
    if y_true.sum() == 0:  
        if y_pred.sum() == 0:  
            return 1  
        else:  
            return 0  

    return jaccard(y_true, y_pred)  


def calculate_auc(y_true, y_pred):  
    if (y_true.sum() == 0) and (y_pred.sum() == 0):  
        return 1.0
    elif (y_true.sum() == 0) and (y_pred.sum() > 0):  
        return 0.0
    elif (y_true.sum() > 0) and (y_pred.sum() == 0):  
        return 0.0
    else:  
        return roc_auc_score(y_true.ravel(), y_pred.ravel()) 


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    arg = parser.add_argument  
    arg('-ground_truth_dir', type=str, required=False, help='path where ground truth images are located',  
        default='./datasets/test/masks')  
    arg('-pred_dir', type=str, required=False, help='path with predictions', default='./pred_results')  
    arg('-threshold', type=float, default=0.3, required=False, help='crack threshold detection')  
    args = parser.parse_args()  

    result_dice = []  
    result_jaccard = []  
    result_iou = []  
    result_auc = []  

    pred_paths = [path for path in Path(args.pred_dir).glob('*')]  
    for pred_file_name in tqdm(pred_paths):  
        pred_image = cv2.imread(str(pred_file_name), 0)  

        if pred_image is None:  
            print(f"Warning: Failed to read prediction image: {pred_file_name}")  
            continue  

        y_pred = (pred_image > 255 * args.threshold).astype(np.uint8)  

        mask_file_name = None  
        for ext in ['.jpg', '.png']:  
            mask_file_path = Path(args.ground_truth_dir) / pred_file_name.with_suffix(ext).name  
            if mask_file_path.exists():  
                mask_file_name = mask_file_path  
                break  

        if mask_file_name is None:  
            print(f"Warning: No corresponding mask found for {pred_file_name.name}")  
            continue  

        y_true = (cv2.imread(str(mask_file_name), 0) > 0).astype(np.uint8)  

        # Reshape the prediction to match the shape of the true mask  
        y_pred = cv2.resize(y_pred, y_true.shape[::-1], interpolation=cv2.INTER_NEAREST)  

        jaccard_value = jaccard(y_true, y_pred)  
        result_jaccard.append(jaccard_value)  

        dice_value = dice(y_true, y_pred)  
        result_dice.append(dice_value)  

        iou_value = jaccard_value  
        result_iou.append(iou_value)  

        auc_value = calculate_auc(y_true, y_pred)  
        result_auc.append(auc_value)  
        result_auc.append(auc_value)  

        # print(f"Image: {pred_file_name.name}")
        # print(f"Jaccard: {jaccard_value:.4f}")
        # print(f"Dice: {dice_value:.4f}")
        # print(f"IoU: {iou_value:.4f}")
        # print(f"AUC: {auc_value:.4f}")
        # print("---")

    print(f"Average Jaccard: {np.mean(result_jaccard):.4f} ± {np.std(result_jaccard):.4f}")  
    print(f"Average Dice: {np.mean(result_dice):.4f} ± {np.std(result_dice):.4f}")  
    print(f"Average IoU: {np.mean(result_iou):.4f} ± {np.std(result_iou):.4f}")  
    print(f"Average AUC: {np.mean(result_auc):.4f} ± {np.std(result_auc):.4f}")