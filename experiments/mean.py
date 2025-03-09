"""Mean."""
import cv2 as cv
import numpy as np

from utils import ImageLoader

imageDataset = '/home/joseph/Documents/Thesis/Dataset/ssivDataset/images'
imageLength = 372
image_offset = 200


def image_mean():
    imshape = load_image(0).shape
    image_acumulator = np.zeros(imshape, dtype=np.int64)
    for i in range(imageLength):
        image_acumulator += load_image(i)
    return (image_acumulator / imageLength).astype(np.uint8)


def main():
    image_loader = ImageLoader(imageDataset, image_offset)
    imMean = image_mean()
    cv.imshow('image mean', imMean)
    # for i in range(3):
    #     cv.imshow(f'frame: {i:03}', load_image(i) - imMean)
    im = image_loader.load(0)
    for i in range(10, 100, 20):
        s = (imMean*(i/100)).astype(np.uint8)
        cv.imshow(f'k = {i:02}%', im-s)

    cv.imshow('original image', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
