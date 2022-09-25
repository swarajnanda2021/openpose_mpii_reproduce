
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


# load mpii dataset that you use saved as a csv file
dataset_mpii = pd.read_csv('../mpii/dataset.csv')

def gaussiankernel1D(x,x_p,sigma): # all entries will be pixel level resolution
    # Gaussian kernel: np.exp(np.abs(x - x_p)**2)/sigma**2
    kernel = np.exp(-np.linalg.norm(x - x_p)**2/sigma**2)
    return kernel

def gaussiankernel2D(halfwidth=3,spread=5):    
    x_space =  np.linspace(-halfwidth,halfwidth,2*halfwidth+1)
    kernel = np.zeros_like(x_space)
    for i in range(np.size(x_space,0)):
        kernel[i] =   gaussiankernel1D(x_space[i],np.array([0, 0]),spread)     
    kernel=np.expand_dims(kernel, axis = 0)
    TwoDkernel = kernel*(kernel.T)
    return TwoDkernel#, plt.imshow(TwoDkernel)
    
def generateHeatmap1Joint(imgsize=[1024,1024],posx=1000,posy=600,kernelhalfwidth=100,spread=30):  
    
    img = np.zeros((imgsize[1], imgsize[0]), dtype=float)
    # Recondition joint position vector
    px = int(kernelhalfwidth)+posx
    py = int(kernelhalfwidth)+posy
    # Make a 2D gaussian kernel 
    mykernel = gaussiankernel2D(kernelhalfwidth,spread)
    # Calculate padding value
    padx = int((np.size(mykernel,0)-1))
    pady = int((np.size(mykernel,1)-1))
    # Pad the image and store separately
    A = np.pad(img,(padx, pady), 'constant', constant_values = (0,0))
    # Merge kernel into empty plot    
    A[px:px+mykernel.shape[0], py:py+mykernel.shape[1]] += mykernel
    # Perform a center crop
    A = A[padx:img.shape[0]+padx, pady:img.shape[1]+pady]
    
    return A


def generateIgnoreMask(imgsize, bbox):
    # pre allocate
    ignoremask = np.zeros((imgsize[1],imgsize[0],3))
    # set roi to 1 based on information from bbox array
    ignoremask[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),0:3] = 1
    
    return ignoremask
    

def generatePAF(imgsize,keypoints,partconnections):
    
    paf = 0    
    
    return paf


#%% Cell for performing data augmentation to an image and corresponding keypoints

#%% Cell for inspecting a particular image

# define a function that inspects the images and generates heatmap, part affinity fields and person mask

def imag(number):
    
    data_sub1 = dataset_mpii.loc[dataset_mpii['ImgNumber'] == number]                       # Get rows with particular value
    if np.size(data_sub1)==0:
        return print('Bad data')
    
    img = data_sub1.iloc[0,0]
    im = Image.open('../mpii/images/'+img) 
    
    nPersons = int(data_sub1.iloc[0,1])
    
    
    heatmap = np.zeros((im.size[1], im.size[0]), dtype=float)
    
    for person in range(nPersons):
        
        
        # Generate mask
        keypoints = np.array(data_sub1.iloc[person,5:33])
        
        traintrue = data_sub1.iloc[person,3]
        
        
        
        # collect masking data Estimate ignore_mask
        bbox = np.array(data_sub1.iloc[person,36:40])
        
        
        
        print(im.size)
        
        mask = generateIgnoreMask(im.size,bbox)
        
        
        print(np.shape(mask))
        
        
        maskedimg = im*mask/255
        
        plt.imshow(maskedimg)
        plt.show()
        
        
        
        
        print(person)
        implot = plt.imshow(im)
        # plot keypoints
        px = 4
        while px <36:        
            # plot keypoints using your data
            x = data_sub1.iloc[person,px]
            y = data_sub1.iloc[person,px+1]
            px=px+2
            if x<0 and y<0:
                continue
            print(x,y)
            print(y,np.size(im,1))
            if x<np.size(im,0) or y<np.size(im,1):
                #print(col,x,y)
                plt.scatter([[x]],[[y]])
                # make heatmaps
                heatmap += generateHeatmap1Joint(imgsize=im.size,posx=int(y),posy=int(x),kernelhalfwidth=200,spread=30)
        plt.show()
    
    implot2 = plt.imshow(heatmap)
    plt.colorbar()
    plt.show()
    
    return maskedimg,data_sub1
    




   
    
    

if __name__ == '__main__':
    
    imag(129)
