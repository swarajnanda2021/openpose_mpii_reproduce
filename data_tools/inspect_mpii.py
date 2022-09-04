
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

# define a 1d Gaussian kernel from the paper
def gaussiankernel1D(x,x_p,sigma): # all entries will be pixel level resolution
    # Gaussian kernel: np.exp(np.abs(x - x_p)**2)/sigma**2
    kernel = np.exp(-np.linalg.norm(x - x_p)**2/sigma**2)
    return kernel
  
# generate a 2D gaussian kernel for the keypoint 
def gaussiankernel2D(halfwidth=3,spread=5):    
    x_space =  np.linspace(-halfwidth,halfwidth,2*halfwidth+1)
    kernel = np.zeros_like(x_space)
    for i in range(np.size(x_space,0)):
        kernel[i] =   gaussiankernel1D(x_space[i],np.array([0, 0]),spread)     
    kernel=np.expand_dims(kernel, axis = 0)
    TwoDkernel = kernel*(kernel.T)
    return TwoDkernel#, plt.imshow(TwoDkernel)

# generate a heatmap for a single keypoint    
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

# generate the PAF of the limbs
def generatePAF(imgsize,links):
    # work in prog
    img = np.zeros((imgsize[1], imgsize[0]), dtype=float)
    
    
    
    
    
    return img




# define a function that inspects the images and generates heatmap and part affinity fields
def imag(number):
    #img = datfrm['NAME'][number]
    #print(img)
    data_sub1 = dataset_mpii.loc[dataset_mpii['ImgNumber'] == number]  # Get rows with particular value
    if np.size(data_sub1)==0: # do not plot image numbers that have no data (super strange)
        return print('Bad data')
    
    
    print(data_sub1) 
    img = data_sub1.iloc[0,1]
    
    
    traintrue = data_sub1.iloc[0,4]
    print(traintrue)
    
    nPersons = data_sub1.iloc[0,2]
    
    im = Image.open('../mpii/images/'+img) 
    print(im.size)
    
    
    
    heatmap = np.zeros((im.size[1], im.size[0]), dtype=float)
    for person in range(0,data_sub1.shape[0]): # loop over people
        print(person)
        implot = plt.imshow(im)
        # plot keypoints
        px = 5
        while px <37:        
            
            # plot keypoints using your data
            x = data_sub1.iloc[person,px]
            y = data_sub1.iloc[person,px+1]
            px=px+2
            
            if x<0 and y<0: # do not plot the -1s
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
    
    
    

if __name__ == '__main__':
    # run this to plot and see the heatmaps and keypoint data
    imag(10)
