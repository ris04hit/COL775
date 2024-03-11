# Imports
import cv2
import numpy as np
import os
import pickle
import torch

# Creating directory
def create_dir(addr):
    if not os.path.exists(addr):
        os.mkdir(addr)

# Partition data into smaller fragments and returns sum and quad sum
def moment_data(category, shape, address, stat_addr, mode):
    # Reading Data and Preprocessing it
    lin_sum, quad_sum = np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)
    ct_list = []
    for cat in category:
        cat_path = os.path.join(address, cat)
        
        # Checking if already present
        if os.path.exists(stat_addr) or mode==1:
            ct_list.append(len(os.listdir(cat_path)))
            continue
        
        # Reading each image of particular category
        count = 0
        for img_name in sorted(os.listdir(cat_path)):
            img_path = os.path.join(cat_path, img_name)
            img = np.array(cv2.imread(img_path))
            lin_sum += img.astype(np.float64)
            quad_sum += np.square(img.astype(np.float64))
            count += 1
        ct_list.append(count)
        
        # log
        print(f"Read {cat}\tindex: {category.index(cat)}\tnum: {count}")

    if mode == 0:
        print("Read Train Data")
    else:
        print("Read Test Data")

    return lin_sum, quad_sum, ct_list

# Loading statistical data
def load_stat(stat_addr):
    x_stats = np.load(stat_addr)
    x_mean = x_stats['mean']
    x_std = x_stats['std']
    return x_mean, x_std

# Saving statistical data
def save_stat(lin_sum, quad_sum, stat_addr, total_ct):
    x_mean = (lin_sum/total_ct).astype(np.float64)
    x_std = np.sqrt(quad_sum/total_ct - x_mean**2).astype(np.float64)
    np.savez_compressed(stat_addr, mean = x_mean, std = x_std)
    print("Saved statistical processed data")

# Reading data fragments and normalizing them
def normalize(addr, category, processed_x_addr, processed_y_addr, norm, overwrite):
    ct = 0
    for cat in category:
        cat_addr = os.path.join(addr, cat)
        for img_name in sorted(os.listdir(cat_addr)):
            # Save Address
            x_addr = os.path.join(processed_x_addr, f'{ct}.pt')
            y_addr = os.path.join(processed_y_addr, f'{ct}.pt')
            if not overwrite and os.path.exists(x_addr) and os.path.exists(y_addr):
                ct += 1
                continue
            
            # Reading image
            img_path = os.path.join(cat_addr, img_name)
            img = np.array(cv2.imread(img_path), dtype=np.float64)
            
            # Normalizing image
            norm_img = norm(img)
            x = torch.tensor(norm_img)
            
            # Getting y
            y = np.zeros((len(category),), dtype=np.float64)
            y[category.index(cat)] = 1
            y = torch.Tensor(y)
            
            # Saving image
            torch.save(x, x_addr)
            torch.save(y, y_addr)
            
            # Incrementing counter
            ct += 1
        
        print(f"Preprocessed\tindex: {category.index(cat)}\t{cat}")

# Preprocessed data
def preprocess(address, mode, overwrite = False, cat_addr = None, stat_addr = None, processed_x_addr = None, processed_y_addr = None, shape = (256, 256, 3)):
    '''
    mode = 0: Training Mode
    mode = 1: Testing Mode
    '''

    if mode not in [0, 1]:
        raise Exception("Not a Valid Mode")

    # Identifying Categories
    if mode == 0:
        category = sorted(os.listdir(address))
        
        # Saving categories
        if overwrite or not os.path.exists(cat_addr):
            with open(cat_addr, 'wb') as cat_file:
                pickle.dump(category, cat_file)
        print("Assigned Categories")
        
    elif mode == 1:
        # Reading Category array
        with open(cat_addr, 'rb') as cat_file:
            category = pickle.load(cat_file)
        print("Read Categories")
    
    # count of images and their moments
    lin_sum, quad_sum, ct_list = moment_data(category=category, shape=shape, address=address, stat_addr=stat_addr, mode=mode)
        
    # Normalizing Data
    if mode == 0 and (overwrite or not os.path.exists(stat_addr)):
        save_stat(lin_sum=lin_sum, quad_sum=quad_sum, stat_addr=stat_addr, total_ct=sum(ct_list))
    x_mean, x_std = load_stat(stat_addr)
        
    # Function for normalization
    norm = lambda img: np.divide((img - x_mean), x_std, out = np.zeros_like(x_mean), where = x_std!=0)

    # Creating train directory
    create_dir(processed_x_addr)
    create_dir(processed_y_addr)

    # Reading Saved Files and Normalizing them
    normalize(addr=address, category=category, processed_x_addr=processed_x_addr, processed_y_addr=processed_y_addr, norm=norm, overwrite=overwrite)

