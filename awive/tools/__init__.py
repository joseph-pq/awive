import cv2
from numpy.typing import NDArray

def imshow(img: NDArray, title: str = "Image") -> None:
    if img.shape[0] > img.shape[1]:
        new_height = 512
        new_width = int(img.shape[1] / img.shape[0] * new_height)
    else:
        new_width = 1024
        new_height = int(img.shape[0] / img.shape[1] * new_width)
    img = cv2.resize(img, (new_width, new_height))
    print(f"{img.shape=}")

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
