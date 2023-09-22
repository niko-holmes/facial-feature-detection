#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import *
from typing import List, Tuple, Set, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import cv2

Image = np.ndarray;


# In[ ]:


OUTPUT_DIRECTORY = 'DIRECTORY';
INPUT_DIRECTORY = 'DIRECTORY';


# # Constants

# #### Convolution kernels

# In[ ]:


K_GAUSSIAN3x3 = (1/16) * np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]);


# #### Haar-Like Features

# In[ ]:


# Haar-Like feature dictionary

NOSE = np.array([[0, 1, 0],
                 [1, 1, 1],
                 [0, 0, 0]]);

LIPS = np.array([[1],
                 [0],
                 [1]]);

EYE = np.array([[1, 0, 1]]);

FOREHEAD = np.array([[0],
                     [1]]);

EYEBROWS = np.array([[1],
                     [0]]);

RIGHTFACE = np.array([[1,0]]);

LEFTFACE = np.array([[0,1]]);


# # Import Data 

# In[ ]:


SUBJECTS_N = ['4', '15', '21', '22', '24'];
IMG_NAMES = ["TD_RGB_E_1", "TD_RGB_E_2", "TD_RGB_E_3", "TD_RGB_E_4", "TD_RGB_E_5"];

target_images = [cv2.resize(cv2.imread(INPUT_DIRECTORY + n + '/' + IMG_NAMES[0] + ".jpg", 0), 
                            None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_LINEAR) for n in SUBJECTS_N];

test_imgs_2 = [cv2.resize(cv2.imread(INPUT_DIRECTORY + n + '/' + IMG_NAMES[1] + ".jpg", 0), 
                            None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_LINEAR) for n in SUBJECTS_N];
test_imgs_3 = [cv2.resize(cv2.imread(INPUT_DIRECTORY + n + '/' + IMG_NAMES[2] + ".jpg", 0), 
                            None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_LINEAR) for n in SUBJECTS_N];
test_imgs_4 = [cv2.resize(cv2.imread(INPUT_DIRECTORY + n + '/' + IMG_NAMES[3] + ".jpg", 0), 
                            None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_LINEAR) for n in SUBJECTS_N];
test_imgs_5 = [cv2.resize(cv2.imread(INPUT_DIRECTORY + n + '/' + IMG_NAMES[4] + ".jpg", 0), 
                            None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_LINEAR) for n in SUBJECTS_N];
test_images = test_imgs_2 + test_imgs_3 + test_imgs_4 + test_imgs_5;


# # Functions

# ### Preprocess Data

# In[ ]:


def PreProcess(image: Image, CannyLowThres: int = 25, CannyHighThres: int = 80) -> Image:
    """Applies a 3x3 Gaussian blur followed by Canny edge detection to the input image.
    Returns edge detection result with values in range [0, 1]
    
    :image         : Input image
    :CannyLowThres : Low threshold for hysterisis reduction
    :CannyHighThres: High threshpld for hysterisis reduction
    :return        : Edge detection image
    """
    im_blur = cv2.filter2D(image, -1, K_GAUSSIAN3x3);
    im_edge = cv2.Canny(im_blur, 25, 80);
    #im_thres = np.round(im_edge / 255).astype(int);
    
    return im_edge


# ### Determine Face Area

# In[ ]:


def DetectFaceRect(image: Image, smooths: int = 1, diff_thres: int = 5, chin_padding: float = 0.1, 
                   min_width_thres: float = 0.25, print_output: bool = False) -> List[Tuple[int,int]]:
    """ 
    Assumptions:
    * image is binary with values either 0 or 1
    * image is edge detection image
    
    :image           : Input binary image
    :smooths         : Number of times distance data is smoothed, 0 by default
    :diff_thres      : Threshold for distance difference thesholding, 2 by default
    :chin_padding    : Percentage increase of lower face area bound to include chin, 0.1 by default
    :min_width_thres : Upper face area bound will be placed when face width is >= min_width_thres * max(width), 
                        0.25 by default
    :print_output    : Print graphical output of all steps, default False
    :return          : List containing tuples of face area corner coordinates. Corners are in order NW, NE, SW, SE.
    """

    ## Calculate distance from left side of frame to first edge
    dists = [];
    widths = []
    for y in range(image.shape[0]):
        row_vals = [x for x in range(image.shape[1]) if image[y,x] != 0];
        if row_vals:
            dist = min(row_vals);
            width = max(row_vals) - dist;
        else:
            dist = 0;
            width = 0;
        dists.append(dist);
        widths.append(width);

    #############################################################
    ## Smooth distances
    def smooth(data: List[int], smooth_its: int = 1) -> List[int]:
        """Smooths a data set smooth_its times. Each point is set to the average value between its directly 
        adjacent points. Returns a tuple of smoothed values.

        :data      : data to be smoothed
        :smooth_its: number of times smoothing is applied
        :return    : smoothed data
        """
        if smooth_its > 0:
            axis1 = np.linspace(0, len(data), len(data), dtype = int);
            data_out = data.copy();
            data_out_1 = data_out.copy()

            for n in range(smooth_its):
                for i in range(1, len(axis1)-1):
                    data_out_1[i] = (data_out[i-1] + data_out[i+1]) / 2;
                data_out = data_out_1;

            return data_out;
        else:
            return data;

    dists_smooth = smooth(dists, smooths);

    #############################################################
    ## FIND NECK
    
    # Chamption approach.
    # Start at y = 0, fnd differences between first point and point i until the diff is greater than threshold
    # Then move to the next point (y = 1)
    # Save longest segment
    
    max_segment = [0, 0];
    for y in range(len(dists_smooth) // 2, len(dists_smooth)-1):
        i = 1;
        while (y+i < len(dists_smooth)) and (abs(dists_smooth[y+i] - dists_smooth[y]) < diff_thres):
            i += 1;
        if i-1 >= max_segment[1]:
            max_segment = [y, i-1];
        

    #############################################################
    ## DEFINE FACE RECTANGLE
    face_bottom = max_segment[0] + int(max_segment[0]*chin_padding);                              # Y max
    face_top = min(i for i in range(len(dists[:face_bottom])) 
                   if (dists[i] != 0 and widths[i] > min_width_thres*max(widths[:face_bottom]))); # Y min
    face_left = min(d for d in dists[face_top:face_bottom] if d != 0);                            # X min
    face_right = face_left + max(widths[face_top:face_bottom]);                                   # X max

    face_rect = [(face_left, face_top), (face_right, face_top), (face_left, face_bottom), (face_right, face_bottom)];

    #############################################################
    if print_output:
        plt.figure();
        plt.plot(dists, np.linspace(0, len(dists), len(dists), dtype=int));
        plt.plot([0, max(dists)], [face_bottom, face_bottom], c='g');
        plt.plot([0, max(dists)], [face_top, face_top], c='g');
        plt.title("Distance");
        ax = plt.gca();
        ax.invert_yaxis();
        
        plt.figure();
        plt.plot(dists_smooth, np.linspace(0, len(dists_smooth), len(dists_smooth), dtype=int));
        plt.plot([0, max(dists_smooth)], [face_bottom, face_bottom], c='g');
        plt.plot([0, max(dists_smooth)], [face_top, face_top], c='g');
        plt.title("Smoothed Distance");
        ax = plt.gca();
        ax.invert_yaxis();

        plt.figure();
        plt.plot(widths[face_top:face_bottom], np.linspace(0, len(widths[face_top:face_bottom]), 
                                                           len(widths[face_top:face_bottom]), dtype = int));
        plt.title("Width");
        ax = plt.gca();
        ax.invert_yaxis();
    #############################################################
    return face_rect;


# ### Haar-Like Feature Detection Functions

# In[ ]:


def HaarLikeFeature(feature: np.array, rect: List[Tuple[int, int]], fr_int: Image) -> float:
    """ 
    :feature: Haar-like feature to be used in evaluation
    :rect   : Size of Haar-like feature
    :fr_int : Integral image of image being evaluated
    :return : Haar-like feature value, the larger the value the 
    """
    ## DEFINE RECTANGLE COORDINATES FOR HAAR-LIKE FEATURE

    shape = feature.shape;
    Xdivs = np.linspace(rect[0][0], rect[1][0], shape[1]+1, dtype = int);
    Ydivs = np.linspace(rect[0][1], rect[2][1], shape[0]+1, dtype = int);

    white_rects = [];
    black_rects = [];

    for rect_x in range(shape[1]):
        for rect_y in range(shape[0]):
            xmin, xmax = Xdivs[rect_x], Xdivs[rect_x + 1];
            ymin, ymax = Ydivs[rect_y], Ydivs[rect_y + 1];
            rect = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)];

            if feature[rect_y, rect_x] == 1:
                # White rectangle
                white_rects.append(rect);
            elif feature[rect_y, rect_x] == 0:
                # Black rectangle
                black_rects.append(rect);

    ############################################################
    ## CALCULATE VALUE OF HAAR-LIKE FEATURE
    # Sum of black square
    sum_black = 0;
    for r in black_rects:
        pNW = fr_int[r[0][1], r[0][0]];
        pNE = fr_int[r[1][1], r[1][0]];
        pSW = fr_int[r[2][1], r[2][0]];
        pSE = fr_int[r[3][1], r[3][0]];
        sum_black += (pSE - pNE - pSW + pNW); 

    # Sum of white square
    sum_white = 0;
    for r in white_rects:
        pNW = fr_int[r[0][1], r[0][0]];
        pNE = fr_int[r[1][1], r[1][0]];
        pSW = fr_int[r[2][1], r[2][0]];
        pSE = fr_int[r[3][1], r[3][0]];
        sum_white += (pSE - pNE - pSW + pNW);

    if sum_black > sum_white:
        return 0;
    else:
        haar_val = (sum_black) - (sum_white);
        return abs(haar_val);


# ### Refine Face Boundary

# In[ ]:


def RefineFaceRect(image: Image, boundary: List[Tuple[int, int]], fr_int: Image) -> List[Tuple[int, int]]:
    """Uses Haar-like features to locate the position of the forehead and cheeks of a face in a given unrefined boundary.
    Refines the boundary to be located at the forehead and cheek positions.
    
    :image   : Input image
    :boundary: Unrefined face boundary
    :fr_int  : Integral image of image being evaluated
    :return  : Refined face boundary
    """

    ## EVALUATE FOREHEAD
    fhead_w = boundary[1][0] - boundary[0][0];
    fhead_h = 50;

    best_fhead = [];

    x_fh = boundary[0][0];
    for y in range(boundary[1][1], boundary[1][1] + int((0.5)*(boundary[2][1]-boundary[1][1])) - fhead_h):
        rect = [(x_fh, y), (x_fh + fhead_w, y), (x_fh, y + fhead_h), (x_fh + fhead_w, y + fhead_h)];

        fhead_val = HaarLikeFeature(FOREHEAD, rect, fr_int);
        if not best_fhead:
            best_fhead = [rect, fhead_val]
        elif fhead_val > best_fhead[1]:
            best_fhead = [rect, fhead_val];

    fhead_rect_medianY = (best_fhead[0][2][1] - best_fhead[0][1][1])//2;
    new_face_top = best_fhead[0][1][1] + fhead_rect_medianY;

    boundary = [(boundary[0][0], new_face_top), (boundary[1][0], new_face_top), boundary[2], boundary[3]];

    #############################################################
    ## EVALUATE RIGHT FACE
    Rface_w = 50;
    Rface_h = boundary[2][1] - boundary[1][1];

    best_Rface = [];

    y_rf = boundary[1][1];
    for x in range(int(0.25*(boundary[1][0] - boundary[0][0])), boundary[1][0] - Rface_w):
        rect = [(x, y_rf), (x + Rface_w, y_rf), (x, y_rf + Rface_h), (x + Rface_w, y_rf + Rface_h)];

        Rface_val = HaarLikeFeature(RIGHTFACE, rect, fr_int);
        if not best_Rface:
            best_Rface = [rect, Rface_val]
        elif Rface_val > best_Rface[1]:
            best_Rface = [rect, Rface_val];

    Rface_rect_medianX = (best_Rface[0][1][0] - best_Rface[0][0][0])//2;
    new_face_right = best_Rface[0][0][0] + Rface_rect_medianX;

    boundary = [boundary[0], (new_face_right, boundary[1][1]), boundary[2], (new_face_right, boundary[3][1])];

    #############################################################
    ## EVALUATE RIGHT FACE
    Lface_w = 50;
    Lface_h = boundary[2][1] - boundary[1][1];

    best_Lface = [];

    y_lf = boundary[1][1];
    for x in range(boundary[0][0], boundary[0][0] + int(0.75*(boundary[1][0] - boundary[0][0])) - Lface_w):
        rect = [(x, y_lf), (x + Lface_w, y_lf), (x, y_lf + Lface_h), (x + Lface_w, y_lf + Lface_h)];

        Lface_val = HaarLikeFeature(LEFTFACE, rect, fr_int);
        if not best_Lface:
            best_Lface = [rect, Lface_val]
        elif Lface_val > best_Lface[1]:
            best_Lface = [rect, Lface_val];

    Lface_rect_medianX = (best_Lface[0][1][0] - best_Lface[0][0][0])//2;
    new_face_left = best_Lface[0][0][0] + Lface_rect_medianX;

    boundary = [(new_face_left, boundary[0][1]), boundary[1], (new_face_left, boundary[2][1]), boundary[3]];

    return boundary;


# ### Detect Facial Features

# In[ ]:


def DetectFeatures(boundary: List[Tuple[int, int]], fr_int: Image) -> Dict[str, List[List[Tuple[int,int]]]]:
    """Determines the bounding boxes for the following facial features: Nose, Lips, Eyes, Eyebrows
    Returns a dictionary of bounding boxes.
    
    :boundary: Face bounding box
    :fr_int  : Integral image of image being evaluated
    :return  : Dictionary listing bounding box of each feature
    """

    best_nose = [];
    best_lips = [];
    best_eyebr = [];
    best_Leye = [];
    best_Reye = [];

    #############################################################
    ## EVALUATE NOSE
    print("Evaluating Nose");

    nose_w = int((1/5)*(boundary[1][0] - boundary[0][0]));
    nose_h = int((1/3)*(boundary[2][1] - boundary[1][1]));

    x = boundary[0][0] + 2*nose_w;
    for y in range(boundary[1][1], boundary[2][1] - nose_h):    
        rect = [(int(0.98*x), y), 
                (int(1.02*(x + nose_w)), y), 
                (int(0.98*x), y + nose_h), 
                (int(1.02*(x + nose_w)), y + nose_h)];

        nose_val = HaarLikeFeature(NOSE, rect, fr_int);
        if not best_nose:
            best_nose = [rect, nose_val]
        elif nose_val > best_nose[1]:
            best_nose = [rect, nose_val];

    #############################################################
    ## EVALUATE LIPS
    print("Evaluating Lips");

    lips_w = int((2/5)*(boundary[1][0] - boundary[0][0]));
    lips_h = int((1/9)*(boundary[2][1] - boundary[1][1]));

    x = boundary[0][0] + int((1.5/5)*(boundary[1][0] - boundary[0][0]));
    for y in range(boundary[1][1] + int((2/3)*(boundary[2][1] - boundary[1][1])), boundary[2][1] - lips_h):
        rect = [(x, y), (x + lips_w, y), (x, y + lips_h), (x + lips_w, y + lips_h)];

        lips_val = HaarLikeFeature(LIPS, rect, fr_int);
        if not best_lips:
            best_lips = [rect, lips_val];
        elif lips_val > best_lips[1]:
            best_lips = [rect, lips_val];

    #############################################################
    ## EVALUATE EYEBROWS

    eyebr_w = boundary[1][0] - boundary[0][0];
    eyebr_h = 20;

    print("Evaluating Eyebrows");
    x = boundary[0][0];
    for y in range(boundary[1][1], boundary[2][1] - eyebr_h):
        rect = [(x, y), (x + eyebr_w, y), (x, y + eyebr_h), (x + eyebr_w, y + eyebr_h)];

        eyebr_val = HaarLikeFeature(EYEBROWS, rect, fr_int);
        if not best_eyebr:
            best_eyebr = [rect, eyebr_val];
        elif eyebr_val > best_eyebr[1]:
            best_eyebr = [rect, eyebr_val];

    #############################################################
    ## EVALUATE LEFT EYE

    eye_w = int((1/5)*(boundary[1][0] - boundary[0][0]));
    eye_h = int((1/6)*(boundary[2][1] - boundary[1][1]));

    print("Evaluating Left Eye");
    y = best_eyebr[0][2][1];
    for x in range(boundary[0][0], best_nose[0][0][0] - eye_w):    
        rect = [(x, y), (x + eye_w, y), (x, y + eye_h), (x + eye_w, y + eye_h)];

        eye_val = HaarLikeFeature(EYE, rect, fr_int);
        if not best_Leye:
            best_Leye = [rect, eye_val];
        elif eye_val > best_Leye[1]:
            best_Leye = [rect, eye_val];

    #############################################################
    ## EVALUATE RIGHT EYE

    eye_w = int((1/5)*(boundary[1][0] - boundary[0][0]));
    eye_h = int((1/6)*(boundary[2][1] - boundary[1][1]));

    print("Evaluating Right Eye");
    y = best_eyebr[0][2][1];
    for x in range(best_nose[0][1][0], boundary[1][0] - eye_w):    
        rect = [(x, y), (x + eye_w, y), (x, y + eye_h), (x + eye_w, y + eye_h)];

        eye_val = HaarLikeFeature(EYE, rect, fr_int);
        if not best_Reye:
            best_Reye = [rect, eye_val];
        elif eye_val > best_Reye[1]:
            best_Reye = [rect, eye_val];

    print("Done");
    #############################################################
    
    return {"Nose": best_nose[0], "Lips": best_lips[0], "Eye_R": best_Reye[0], "Eye_L": best_Leye[0], 
            "Eyebrows": best_eyebr[0]};


# # Evaluate all Images

# In[ ]:


target_features = [];

i = 1;

fig, ax = plt.subplots(1, 2, figsize = (7,7), sharey = True);
for image in target_images[0:2]:
    img_thres = PreProcess(image);
    boundary_unref = DetectFaceRect(img_thres);
    fr_int = cv2.integral(image); # Calculate Integral of image
    boundary_ref = RefineFaceRect(image, boundary_unref, fr_int);
    feat_dict = DetectFeatures(boundary_ref, fr_int);
    target_features.append(feat_dict);

    
    ax[i-1].imshow(image, cmap = 'gray');
    ax[i-1].add_patch(pch.Rectangle(feat_dict["Nose"][0], feat_dict["Nose"][1][0] - feat_dict["Nose"][0][0], 
                               feat_dict["Nose"][2][1] - feat_dict["Nose"][0][1], linewidth = 1, 
                               edgecolor = 'r', facecolor = 'None'));
    ax[i-1].add_patch(pch.Rectangle(feat_dict["Lips"][0], feat_dict["Lips"][1][0] - feat_dict["Lips"][0][0], 
                               feat_dict["Lips"][2][1] - feat_dict["Lips"][0][1], linewidth = 1, 
                               edgecolor = 'b', facecolor = 'None'));
    ax[i-1].add_patch(pch.Rectangle(feat_dict["Eyebrows"][0], feat_dict["Eyebrows"][1][0] - feat_dict["Eyebrows"][0][0], 
                               feat_dict["Eyebrows"][2][1] - feat_dict["Eyebrows"][0][1], linewidth = 1, 
                               edgecolor = 'purple', facecolor = 'None'));
    ax[i-1].add_patch(pch.Rectangle(feat_dict["Eye_R"][0], feat_dict["Eye_R"][1][0] - feat_dict["Eye_R"][0][0], 
                               feat_dict["Eye_R"][2][1] - feat_dict["Eye_R"][0][1], linewidth = 1, 
                               edgecolor = 'y', facecolor = 'None'));
    ax[i-1].add_patch(pch.Rectangle(feat_dict["Eye_L"][0], feat_dict["Eye_L"][1][0] - feat_dict["Eye_L"][0][0], 
                               feat_dict["Eye_L"][2][1] - feat_dict["Eye_L"][0][1], linewidth = 1, 
                               edgecolor = 'y', facecolor = 'None'));
    ax[i-1].add_patch(pch.Rectangle(boundary_ref[0], boundary_ref[1][0] - boundary_ref[0][0], 
                               boundary_ref[2][1] - boundary_ref[0][1], linewidth = 1, 
                               edgecolor = 'g', facecolor = 'None')); 
    ax[i-1].set_title(f"Subject {SUBJECTS_N[i-1]}");
    i += 1;

fig.tight_layout();
plt.savefig(OUTPUT_DIRECTORY + f"target_image{i}.jpg");
plt.show()


# In[ ]:


test_features = [];

i = 1;
for image in test_images:
    img_thres = PreProcess(image);
    boundary_unref = DetectFaceRect(img_thres);
    fr_int = cv2.integral(image); # Calculate Integral of image
    boundary_ref = RefineFaceRect(image, boundary_unref, fr_int);
    feat_dict = DetectFeatures(boundary_ref, fr_int);
    target_features.append(feat_dict);

    fig, ax = plt.subplots(1);
    ax.imshow(image, cmap = 'gray');
    ax.add_patch(pch.Rectangle(feat_dict["Nose"][0], feat_dict["Nose"][1][0] - feat_dict["Nose"][0][0], 
                               feat_dict["Nose"][2][1] - feat_dict["Nose"][0][1], linewidth = 1, 
                               edgecolor = 'r', facecolor = 'None'));
    ax.add_patch(pch.Rectangle(feat_dict["Lips"][0], feat_dict["Lips"][1][0] - feat_dict["Lips"][0][0], 
                               feat_dict["Lips"][2][1] - feat_dict["Lips"][0][1], linewidth = 1, 
                               edgecolor = 'b', facecolor = 'None'));
    ax.add_patch(pch.Rectangle(feat_dict["Eyebrows"][0], feat_dict["Eyebrows"][1][0] - feat_dict["Eyebrows"][0][0], 
                               feat_dict["Eyebrows"][2][1] - feat_dict["Eyebrows"][0][1], linewidth = 1, 
                               edgecolor = 'purple', facecolor = 'None'));
    ax.add_patch(pch.Rectangle(feat_dict["Eye_R"][0], feat_dict["Eye_R"][1][0] - feat_dict["Eye_R"][0][0], 
                               feat_dict["Eye_R"][2][1] - feat_dict["Eye_R"][0][1], linewidth = 1, 
                               edgecolor = 'y', facecolor = 'None'));
    ax.add_patch(pch.Rectangle(feat_dict["Eye_L"][0], feat_dict["Eye_L"][1][0] - feat_dict["Eye_L"][0][0], 
                               feat_dict["Eye_L"][2][1] - feat_dict["Eye_L"][0][1], linewidth = 1, 
                               edgecolor = 'y', facecolor = 'None'));
    ax.add_patch(pch.Rectangle(boundary_ref[0], boundary_ref[1][0] - boundary_ref[0][0], 
                               boundary_ref[2][1] - boundary_ref[0][1], linewidth = 1, 
                               edgecolor = 'g', facecolor = 'None'));
    plt.savefig(OUTPUT_DIRECTORY + f"test_image{i}.jpg");
    plt.close();
    
    i += 1;

