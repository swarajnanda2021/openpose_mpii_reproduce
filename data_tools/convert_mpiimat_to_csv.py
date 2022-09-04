import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import scipy.io as spio
import csv

# Some basic book entries for self
body_part = [0, #- r ankle,
             1, # - r knee,
             2, # - r hip,
             3, # - l hip,
             4, # - l knee,
             5, # - l ankle,
             6, # - pelvis,
             7, # - thorax,
             8, # - upper neck,
             9, # - head top,
             10, # - r wrist,
             11, # - r elbow,
             12, # - r shoulder,
             13, # - l shoulder,
             14, # - l elbow,
             15] # - l wrist


limbs = [(0,1), # r calf
         (1,2), # r thign
         (2,6), # r waist
         (4,5), # l calf
         (3,4), # l thign
         (6,3), # l waist
         (7,6), # spine
         (7,13), # l collar
         (7,12), # r collar
         (7,8), # neck
         (8,9), # head
         (12,11), # r bicep
         (13,14), # l bicep
         (11,10), # r arm
         ]

datalist = {} # create a dictionary to store our data



def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return data




datfrm = pd.read_csv('../mpii/data.csv')
# Convert MPII mat dataset to csv file to read
annotations = loadmat('../mpii/labels/mpii_human_pose_v1_u12_1.mat')
release = annotations['RELEASE']
train_test_classifier = release.__dict__['img_train'] # 1 for train, 0 for test
annot_data = release.__dict__['annolist']
nPersons = np.zeros(len(annot_data)) # pre assign nPersons array for reference



#with open(r'../mpii/dataset.csv', 'w',newline='') as fp:
#    writer = csv.writer(fp)    
case_list = []
counter = 0
for i in range(len(annot_data)): # loop over total images
   
    print(i)
    imgname = annot_data[i].image.name
    nPersons[i] = np.size(annot_data[i].annorect) 
    
    for j in range(np.size(annot_data[i].annorect)): # loop over people
        
        
        datalist[counter,0] = imgname
        datalist[counter,1] = nPersons[i]
    
        # Create line entries in the csv file you will write    
        item = (imgname, nPersons[i]) # image info
        pointdata = np.ones((1,32))*-1 # fill -1 for no-show body parts
        if np.size(annot_data[i].annorect) == 1:
            temp_ = annot_data[i].annorect
        else:
            temp_ = annot_data[i].annorect[j]
        if hasattr(temp_,'annopoints') == True and np.size(temp_.annopoints)>0:    # added this for some images that do not have point data in them at all
            # pre-assign points variable    
            for k in range(np.size(temp_.annopoints.point)): # loop over body parts present
            #for k in range(np.size(temp_.annopoints.point)): # loop over body parts present
                
                if np.size(temp_.annopoints.point) > 1:
                    temp_id  = temp_.annopoints.point[k].id
                    temp_x = int(np.ceil(temp_.annopoints.point[k].x))
                    temp_y = int(np.ceil(temp_.annopoints.point[k].y))
                    
                else:
                    temp_id  = temp_.annopoints.point.id
                    temp_x = int(np.ceil(temp_.annopoints.point.x))
                    temp_y = int(np.ceil(temp_.annopoints.point.y))
                    
                pointdata[0,2*temp_id] = temp_x    
                pointdata[0,2*temp_id + 1] = temp_y    
        # Build a case to append to case_list (tedious but should work)                
        case = {"Filename": imgname, 'nPersons': nPersons[i],
                "ImgNumber": i,
                "totrain": train_test_classifier[i], 
                "rankle x": pointdata[0][0],
                "rankle y": pointdata[0][1],
                "rknee x": pointdata[0][2],
                "rknee y": pointdata[0][3],
                "rhip x": pointdata[0][4],
                "rhip y": pointdata[0][5],
                "lhip x": pointdata[0][6],
                "lhip y": pointdata[0][7],
                "lknee x": pointdata[0][8],
                "lknee y": pointdata[0][9],
                "lankle x": pointdata[0][10],
                "lankle y": pointdata[0][11],
                "pelvis x": pointdata[0][12],
                "pelvis y": pointdata[0][13],
                "thorax x": pointdata[0][14],
                "thorax y": pointdata[0][15],
                "upperneck x": pointdata[0][16],
                "upperneck y": pointdata[0][17],
                "headtop x": pointdata[0][18],
                "headtop y": pointdata[0][19],
                "rwrist x": pointdata[0][20],
                "rwrist y": pointdata[0][21],
                "relbow x": pointdata[0][22],
                "relbow y": pointdata[0][23],
                "rshoulder x": pointdata[0][24],
                "rshoulder y": pointdata[0][25],
                "lshoulder x": pointdata[0][26],
                "lshoulder y": pointdata[0][27],
                "lelbow x": pointdata[0][28],
                "lelbow y": pointdata[0][29],
                "lwrist x": pointdata[0][30],
                "lwrist y": pointdata[0][31]
                }
        
        
        
        
        case_list.append(case)       
        
        
dataset_mpii = pd.DataFrame.from_dict(case_list)
dataset_mpii.to_csv('../mpii/dataset.csv')      
