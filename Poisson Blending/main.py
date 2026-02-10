import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import os

from align_target import align_target

SOURCE_PATH = "source1.jpg"
TARGET_PATH = "target.jpg"
RESULT_DIR = "results/"


def check_is_boundary(i, j, target_mask):
    """ check if pixel in on the mask boundary or image boundary """
    height = target_mask.shape[0]
    width = target_mask.shape[1]
    is_image_boundary = i == 0 or i == height - 1 or j == 0 or j == width - 1
    if is_image_boundary:
        return True
    else:
        kernel_indices = np.array([[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]])
        return np.any(target_mask[kernel_indices[:, 0], kernel_indices[:, 1]] == 0)


def get_laplacian_coefficients(target_mask):
    """ Returns coefficient matrix A"""
    pixel_count = int(np.sum(target_mask))
    coefficient = lil_matrix((pixel_count, pixel_count))
    indices = np.argwhere(target_mask != 0)
    index_mapping = {tuple(coord): i for i, coord in enumerate(indices)}

    for pixel_index, (i, j)  in enumerate(indices):
        kernel_indices =[[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]
        is_boundary = check_is_boundary(i, j, target_mask)
        if is_boundary:
            # boundary constrains
            coefficient[pixel_index, pixel_index] = 1
        else:
            coefficient[pixel_index, pixel_index] = -4
            coefficient[
                pixel_index, [index_mapping[(x,y)] for x,y in kernel_indices]
            ] = 1

    return coefficient.tocsr()


def get_b_vector(source_image, target_image, target_mask):
    """ Get right-hand vector b"""
    indices = np.where(target_mask != 0)
    laplacian = cv2.Laplacian(source_image, cv2.CV_64F)
    for i, j in zip(indices[0], indices[1]):
        if check_is_boundary(i, j, target_mask):
            # boundary constrains
            laplacian[i, j] = target_image[i, j]
    return laplacian[indices].reshape(-1,3)


def plot_image(image, error, source_image):
    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Source')
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Result\nError: {error:.15e}")
    ax[1].axis("off")

    fig.tight_layout()
    result_path = RESULT_DIR + SOURCE_PATH.replace(".jpg", "_blended_result.png")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    plt.savefig(result_path, dpi=150, bbox_inches="tight")
    plt.show()


def poisson_blend(source_image, target_image, target_mask):
    # source_image: image to be cloned
    # target_image: image to be cloned into
    # target_mask: mask of the target image
    source_image = source_image.astype(np.float64)
    target_image = target_image.astype(np.float64)

    b = get_b_vector(source_image, target_image, target_mask)
    coefficients = get_laplacian_coefficients(target_mask)
    res = spsolve(coefficients, b)
    error = np.linalg.norm(coefficients.dot(res) - b)
    blended_values = np.clip(res, 0, 255)
    blended_image = target_image.copy()
    blended_image[target_mask != 0] = blended_values
    return blended_image.astype(np.uint8), error


if __name__ == "__main__":
    # read source and target images
    source_path = SOURCE_PATH
    target_path = TARGET_PATH
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    # align target image
    im_source, mask = align_target(source_image, target_image)

    ##poisson blend
    blended_image, error = poisson_blend(im_source, target_image, mask)
    plot_image(blended_image, error, im_source)
    
    # save image
    result_path = RESULT_DIR + SOURCE_PATH.replace('.jpg', '_blended.png')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    cv2.imwrite(result_path,blended_image)
