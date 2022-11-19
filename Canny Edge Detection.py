from PIL import Image # pillow package
import numpy as np
from scipy import ndimage

def read_img_as_array(file):
    # read image and convert it to numpy array with dtype np.float64
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

# RGB to Greyscale  
def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray


def sobel(arr):
    # Apply sobel operator on arr and return the result.
    Gx_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = ndimage.convolve(arr, Gx_kernel)
    Gy = ndimage.convolve(arr, Gy_kernel)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    return G, Gx, Gy

def nonmax_suppress(G, Gx, Gy):
    # Suppress non-max value along direction perpendicular to the edge.
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
   
    angles = np.rad2deg(np.arctan2(Gy, Gx))
    # temp_Gx = Gx / G
    # temp_Gy = Gy / G
    # g = (Gx,Gy)
    # p1 = (ndimage.map_coordinates(G, [i + temp_Gy, j + temp_Gx)]
    # p2 = (ndimage.map_coordinates(G, [i - temp_Gy, j - temp_Gx)]

    size = G.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(G[i, j - 1], G[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(G[i - 1, j - 1], G[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(G[i - 1, j], G[i + 1, j])
            else:
                value_to_compare = max(G[i + 1, j - 1], G[i - 1, j + 1])

            if G[i, j] >= value_to_compare:
                suppressed[i, j] = G[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

def thresholding(G, t):
    # Binarize G according threshold t
    size = G.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if G[i,j] <=t:
                G[i,j] = 0
            else:
                G[i,j] = 255
    return G

def hough(G):
    # Return Hough transform of G
    # TODO: Please complete this function.
    # your code here
    pass


if __name__ == "__main__":
    img = read_img_as_array('./image/road.jpeg')
    #TODO: detect edges on 'img'

    # 1. RGB to Greyscale
    img = read_img_as_array('./image/road.jpeg')

    gray_arr = rgb2gray(img)

    save_array_as_img(gray_arr, './image/gray.jpg')


    # 2. Perform Gaussian smoothing
    gauss = np.array([[1, 2, 1],    # define the gaussian kernel
                    [2, 4, 2],
                    [1, 2, 1]])

    gauss = gauss/gauss.sum()       # normalize the sum to be 1

    gauss_arr = ndimage.convolve(gray_arr, gauss)
    save_array_as_img(gauss_arr, './image/gauss.jpg')


    # 3. Apply sobel operator
    G, Gx, Gy = sobel(gauss_arr)
    save_array_as_img(Gx, './image/G_x.jpg')
    save_array_as_img(Gy, './image/G_y.jpg')
    save_array_as_img(G, './image/G.jpg')

    # 4. Apply Non-maximium supression
    suppressed_G = nonmax_suppress(G, Gx, Gy)
    save_array_as_img(suppressed_G, './image/supress.jpg')

    # 5.Thresholding
    t = 200
    edgemap = thresholding(G, t)
    save_array_as_img(edgemap, './image/edgemap.jpg')


