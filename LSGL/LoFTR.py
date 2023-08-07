''' Realization of feature matching and error calculation.
    Feature matching by calling LoFTR algorithm and mask and Euclidean error calculation for matching results.
'''

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import glob
import os
import shutil
import math
import csv
import pandas as pd
import time
import sys
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
from LSGL.hgmod import delete_mask

'''
    Initialize LoFTR
'''
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("./LSGL/LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

def Unipath(path):
    '''Convert erroneous paths into executable paths.
    Args:
        path(str): Raw processing path for input.
    Returns:
        path(str): Executable paths after processing.
    '''
    path = path.replace('\\', '/')
    return path

def clean(filepath):
    '''Clean up the specified directory.
    Args:
        filepath(str): Paths to be cleared.
    '''
    filepath = filepath[:-1]
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        time.sleep(0.1)
        os.mkdir(filepath)

def outputpath(respath, db_dir):
    '''Creating the output folder path.
    Args:
        respath(str): Parent path of the output folder.
        db_dir(str): Name of the output folder.
    '''
    path = respath + str(db_dir) + '/'
    clean(path)

def avg_score(mconf):
    '''Calculate the average confidence level.
    Args:
        mconf(list): A list that holds the confidence levels of all feature points.
    Returns:
        avg(float): Average confidence level calculation results.
    '''
    total_score = 0
    # Calculate the total confidence level
    for score in mconf:
        total_score = total_score + score
    # Calculate the average confidence level
    avg = total_score / len(mconf)
    return avg

def Euc_distance(mkpts0, mkpts1):
    '''Calculate the Euclidean distance and Euclidean angle.
    Args:
        mkpts0(list): A list with the coordinates of the feature points.
        mkpts1(list): A list with the coordinates of the corresponding feature points.
    Returns:
        RMSE(float): Average Euclidean distance offset results.
    '''
    pre_act = 0
    pre_act_deg = 0
    all_dis = 0
    all_deg = 0
    # Calculate the average Euclidean distance and Euclidean angle
    for i in range(len(mkpts0)):
        poi = mkpts0[i]
        target = mkpts1[i]
        x = poi[0] - target[0]
        y = poi[1] - target[1]
        d = math.sqrt(x * x + y * y)
        degree = math.atan(y / x)
        # Take the first point to confirm the datum offset
        all_dis += d
        all_deg += degree
    # Calculate average length and average angle
    base_distance = all_dis / len(mkpts0)
    base_degree = all_deg / len(mkpts0)
    # sum of squares
    for i in range(len(mkpts0)):
        poi = mkpts0[i]
        target = mkpts1[i]
        x = poi[0] - target[0]
        y = poi[1] - target[1]
        d = math.sqrt(x * x + y * y)
        degree = math.atan(y / x)
        pre_act += pow((d - base_distance), 2)
        pre_act_deg += pow((degree - base_degree), 2)
    # Calculating European distance error results
    RMSE_dis = math.sqrt(pre_act / len(mkpts0))
    RMSE_deg = math.sqrt(pre_act_deg / len(mkpts0))
    # Calculate RMSE
    RMSE = RMSE_dis * RMSE_deg
    return RMSE

def getBestscore(before_score, before_RMSE, after_score, RMSE):
    '''Calculate the best score.
    Args:
        before_score(float):rating for temporary storage.
        before_RMSE(float):Optimal RMSE for temporary storage.
        after_score(float):Scoring for comparison
        RMSE(float):RMSE for comparison
    '''
    if before_score < after_score:
        return after_score, RMSE
    else:
        return before_score, before_RMSE

def match_mask(inpimage):
    '''Matching social media images with mask files.
    Args:
        inpimage(str): Input social media image path.
    Returns:
        maskname(str): Name of the mask file corresponding to the social media image.
    '''
    imagename = inpimage.split(".")[0]
    imagetype = inpimage.split(".")[-1]
    maskname = imagename + "_seg.jpg"
    return maskname

def readmask(mask_images_path, mkpts0, mkpts1, mconf):
    '''Reading a mask file as a mask.
    Args:
        mask_images_path(str): Input social media image path.
        mkpts0(list): A list with the coordinates of the feature points.
        mkpts1(list): A list with the coordinates of the corresponding feature points.
        mconf(list): A list that holds the confidence levels of all feature points.
    Returns:
        mkpts(list): Coordinates of the feature points after the masking process.
    '''
    # Converting a mask image to a matrix
    mask = match_mask(mask_images_path)
    mask = Image.open(mask)
    mask_array = np.array(mask)

    poi = 0
    match_len = len(mkpts0)
    index = []
    while poi < match_len:
        # Retrieve the coordinates of the matching point x,y position
        x = mkpts0[poi][0]
        y = mkpts0[poi][1]
        # Determine if the point is within the mask
        x = int(x)
        y = int(y)
        poirgb = mask_array[y][x]
        if (poirgb == [255, 255, 255]).all():
            # If it is within the mask, the point is recorded
            index.append(poi)
        poi = poi + 1
    # Return processing results
    mkpts0_tmp = np.delete(mkpts0, index, 0)
    mkpts1_tmp = np.delete(mkpts1, index, 0)
    mconf_tmp = np.delete(mconf, index)
    mkpts = [mkpts0_tmp, mkpts1_tmp, mconf_tmp]
    return mkpts

def rating(respath):
    '''Normalized scores.
    Args:
        respath(str): Output path of the result.
    '''
    all_RMSE = 0
    all_score = 0
    # Total statistical score
    for path, dirnames, filenames in os.walk(respath):
        for dir in dirnames:
            numb = dir.split('&')
            all_score += float(numb[-3])
            all_RMSE += float(numb[-2])
    # Display scores on folder names
    for path, dirnames, filenames in os.walk(respath):
        for dir in dirnames:
            dir_raw = dir.split('&')[0]
            numb = dir.split('&')
            try:
                score = float(numb[-3]) / all_score
                RMSE = float(numb[-2]) / all_RMSE
                rate = score / RMSE
            except:
                rate = 0
            os.rename(respath + str(dir) + '/', respath + str(dir_raw) + '_' + str(rate) + '/')

def process(userpath, datapath, respath):
    '''Matching and scoring.
    Args:
        userpath(str): Input social media image path.
        datapath(str): Input Path for Street View Images.
        respath(str): Output path of the matching result.
    '''
    # Build the output folder path
    datapath = datapath + 'downloadPic/'
    respath = respath

    # Clean up the output folder
    clean(respath)

    db_file_path = datapath   # Street View
    mask_file_path = userpath  # Social Media Images

    # Reorganize the path and get the image path
    social_images_path = glob.glob(os.path.join(userpath , '*.jpg')) + glob.glob(os.path.join(userpath, '*.png'))
    social_images_path = delete_mask(social_images_path)
    mask_images_path = glob.glob(os.path.join(mask_file_path , '*_seg.jpg'))
        
    # Starting the Total Match Task Progress Bar
    scale = 150
    print(" Overall social media processing progress ".center(scale // 2,"-"))
    social_images_bar = tqdm(total=len(social_images_path))
    dirpath = []
    db_dirnames = []
    filenames = []
    db_dirnames = os.listdir(db_file_path)
    for j in social_images_path:
        j = Unipath(j)
        # Starts the current match task progress bar
        filename = j.split("/")[-1]
        scale = 100
        print(" Matching: {} ".format(filename).center(scale // 2,"#"))
        db_image_bar = tqdm(total=len(db_dirnames))
        # Traversing the Street View folder
        for db_dir in db_dirnames:
          db_images_path = os.path.join(datapath  , db_dir)
          db_image_path = glob.glob(os.path.join(db_images_path  , '*.jpg')) + glob.glob(os.path.join(db_images_path  , '*.png'))
          # Clean up the output folder
          outputpath(respath, db_dir)

          bestscore = 0
          bestRMSE = 0
          
          for i in db_image_path:

              i = Unipath(i)

              # Reading images
              img0_raw = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
              img1_raw = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
              img0_raw = cv2.resize(img0_raw, (640, 480))
              img1_raw = cv2.resize(img1_raw, (640, 480))

              img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
              img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
              batch = {'image0': img0, 'image1': img1}

              # Matching using the LoFTR algorithm
              with torch.no_grad():
                  matcher(batch)
                  mkpts0 = batch['mkpts0_f'].cpu().numpy()
                  mkpts1 = batch['mkpts1_f'].cpu().numpy()
                  mconf = batch['mconf'].cpu().numpy()

              # Matching results based on mask processing
              mkpts = readmask(j, mkpts0, mkpts1, mconf)
              mkpts0 = mkpts[0]
              mkpts1 = mkpts[1]
              mconf = mkpts[2]

              # If there is a match, return a score, otherwise score 0
              try:
                  score = avg_score(mconf) * len(mkpts0)
                  RMSE = Euc_distance(mkpts0, mkpts1)
              except:
                  score = 0

              # Plotting Matching Results
              color = cm.jet(mconf, alpha=0.7)
              text = [
                  'LoFTR',
                  'Matches: {}'.format(len(mkpts0)),
              ]
              # Recorded Highest score
              bestscore, bestRMSE = getBestscore(bestscore, bestRMSE, score, RMSE)

              # Output the results and record the score with the result filename
              i = i.split('_')[-1]
              i = i.split('.')[0]
              fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1,
                                          path=respath + str(db_dir) + '/' + str(i) + "_" + str(score) + "_" + str(
                                              RMSE) + ".png")

          # Record scores in result folder name
          os.rename(respath + str(db_dir) + '/',
                      respath + str(db_dir) + '&' + str(bestscore) + '&' + str(bestRMSE) + '&' + '/')
          # Close the progress bar
          db_image_bar.update(1)
        db_image_bar.close()
        social_images_bar.update(1)
    social_images_bar.close()
    print('Feature matching task completed!')

def write_res(res_path, res_level):
    '''Export results to CSV.
    Args:
        res_path(str): Output path of the result.
        res_level(int): Level of this match.
    '''
    # Reorganization of CSV paths
    res_level = res_path + str(res_level) + '.csv'
    # Read Match Results
    mult_res = []
    for respath, res_dirnames, resnames in os.walk(res_path):
        for res in res_dirnames:
            res = res.split('_')
            single_res = []
            single_res.append(res[-2])
            single_res.append(res[-1])
            mult_res.append(single_res)
    # Write matches to CSV
    with open(res_level, 'w') as data:
        writer = csv.writer(data)
        header = ['pid', 'res']
        writer.writerow(header)
        writer.writerows(mult_res)

def bestpid(res_level, tmp_path, res_path, roundnum):
    '''Read the optimal result with its coordinates.
    Args:
        res_level(int): Level of this match.
        tmp_path(str): Path to the total matching results.
        res_path(str): Output path of the result.
        roundnum(int): Total level of matches.
    Returns:
        x(str): longitude of best point.
        y(str): latitude of best point.
        best_score(str): Most advantageous ratings.
    '''
    x = 0
    y = 0
    best_score = 0
    # CSV path of the result of a single round of matching
    res_path = res_path + 'csv_files/success.csv'
    # CSV path of result of total matches
    res_level = tmp_path + str(res_level) + '.csv'
    # Read the table of correspondence between PID and point coordinates
    pids = pd.read_csv(res_path, header=0, usecols=['pid', 'wgsx', 'wgsy']).values
    # Read optimal PID
    res = pd.read_csv(res_level, header=0)
    res = res.sort_values(by=['res'], ascending=False)
    res_pid = res.values[roundnum][0]
    res_score = res.values[roundnum][1]
    # Query the coordinates corresponding to the PID
    for pid in pids:
        if pid[2] == res_pid:
            x = pid[0]
            y = pid[1]
            best_score = res_score
    return x, y, best_score
