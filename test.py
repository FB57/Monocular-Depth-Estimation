import numpy as np
import cv2
import matplotlib.pyplot as plt

def variable_blur(im, sigma, ksize=3):
    """Blur an image with a variable Gaussian kernel.
    
    Parameters
    ----------
    im: numpy array, (h, w)
    
    sigma: numpy array, (h, w)
    
    ksize: int
        The box blur kernel size. Should be an odd number >= 3.
        
    Returns
    -------
    im_blurred: numpy array, (h, w)
    
    """
    variance = box_blur_variance(ksize)
    # Number of times to blur per-pixel
    num_box_blurs = 2 * sigma**2 / variance
    # Number of rounds of blurring
    max_blurs = int(np.ceil(np.max(num_box_blurs))) * 3
    # Approximate blurring a variable number of times
    blur_weight = num_box_blurs / max_blurs

    current_im = im
    for i in range(max_blurs):
        next_im = cv2.blur(current_im, (ksize, ksize))
        current_im = next_im * blur_weight + current_im * (1 - blur_weight)
    return current_im

def box_blur_variance(ksize):
    x = np.arange(ksize) - ksize // 2
    x, y = np.meshgrid(x, x)
    return np.mean(x**2 + y**2)

im = np.random.rand(400, 300)
sigma = 3

# Variable
x = np.linspace(0, 1, im.shape[1])
y = np.linspace(0, 1, im.shape[0])
x, y = np.meshgrid(x, y)
print(f"Shape X = {x.shape}, Shape Y = {y.shape}")
sigma_arr = sigma * (x + y)
print(sigma_arr)
im_variable = variable_blur(im, sigma_arr)

# Gaussian
ksize = sigma * 8 + 1
im_gauss = cv2.GaussianBlur(im, (ksize, ksize), sigma)

# Gaussian replica
sigma_arr = np.full_like(im, sigma)
im_approx = variable_blur(im, sigma_arr)

f, axxarr = plt.subplots(1, 2)
f.set_figwidth(16)
f.set_figheight(10)
axxarr[0].imshow(im, cmap="gray")
axxarr[1].imshow(im_variable, cmap="gray")
plt.show()