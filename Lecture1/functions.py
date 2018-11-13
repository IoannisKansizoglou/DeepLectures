import matplotlib.pyplot as plt
from skimage import io, exposure
import pylab, numpy as np

def plot_samples(line=False,theta=[1,1,1],current=[0]):
        
    plt.plot([2,5,7,6,4,8],[1,2,4,5,3,6],'ro')
    plt.plot([1,4,7,5,3,2],[2,5,8,6,6,4],'bo')
    if np.shape(current)[0]>1:
        plt.plot(current[0],current[1],'go')
    if line:
        xaxis = np.array([1,2,3,4,5,6,7,8])
        plt.plot(xaxis, theta[0]/-theta[2]+theta[1]/-theta[2]*xaxis,'black')
    return

def check_accuracy(ind,label):
    if ind == label:
        return 'Σωστή'
    else:
        return 'Λανθασμένη'

def convolve2d(image, kernel):
    
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output
