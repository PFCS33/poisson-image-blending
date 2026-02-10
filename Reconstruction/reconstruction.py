import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import os

SOURCE_PATH = "large1.jpg"
RESULT_DIR = "results/"
AUTO_CORNER = True
CORNER_VALUES = [128, 128, 128, 128]  # top-left, top-right, bottom-left, bottom-right


def get_coefficients(image):
    """Calculate the second derivative coefficients for each pixel

    Returns:
        A matrix (pixel_count, pixel_count)
    """
    height, width = image.shape
    pixel_count = width * height
    coefficients = lil_matrix((pixel_count, pixel_count))
    for i in range(0, height):
        for j in range(0, width):
            row_index = i * width + j
            # detect boundaries
            is_four_corner = (i == 0 or i == height - 1) and (j == 0 or j == width - 1)
            is_horizontal_edge = i == 0 or i == height - 1
            is_vertical_edge = j == 0 or j == width - 1
            if is_four_corner:
                # equal to constant
                coefficients[row_index, row_index] = 1
            elif is_horizontal_edge:
                # only compute horizontal derivative
                coefficients[row_index, [row_index - 1, row_index + 1]] = -1
                coefficients[row_index, row_index] = 2
            elif is_vertical_edge:
                # only compute vertical derivative
                coefficients[row_index, [row_index - width, row_index + width]] = -1
                coefficients[row_index, row_index] = 2
            else:
                coefficients[row_index, [row_index - 1, row_index + 1]] = -1
                coefficients[row_index, [row_index - width, row_index + width]] = -1
                coefficients[row_index, row_index] = 4
    return coefficients.tocsr()


def get_b_vector(source_image, coefficients, corners):
    """Generate b vectors, and add corners constrains"""
    b = coefficients.dot(source_image.flatten())
    _, width = source_image.shape
    pixel_count = coefficients.shape[0]
    b[[0, width - 1, pixel_count - width, pixel_count - 1]] = corners
    return b


def plot_image(origin, target, error):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(origin, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(target, cmap="gray")
    ax[1].set_title(f"Reconstructed\nError: {error:.15e}")
    ax[1].axis("off")

    fig.tight_layout()
    result_path = RESULT_DIR + SOURCE_PATH.replace(".jpg", "_reconstruction_result.png")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    fig.savefig(result_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    ##read source image
    source_image = cv2.imread(SOURCE_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    ##implement reconstruction
    coefficients = get_coefficients(source_image)
    if AUTO_CORNER:
        corners = [
            source_image[0, 0],
            source_image[0, -1],
            source_image[-1, 0],
            source_image[-1, -1],
        ]
    else:
        corners = CORNER_VALUES
    b = get_b_vector(source_image, coefficients, corners)
    print("calculating...")
    res = spsolve(coefficients, b)
    # calculate the error
    error = np.linalg.norm(coefficients.dot(res) - b)
    # plot the result
    reconstructed_image = np.clip(res.reshape(source_image.shape), 0, 255).astype(
        np.uint8
    )
    plot_image(source_image, reconstructed_image, error)
    # save image
    targert_path = RESULT_DIR + SOURCE_PATH.replace(".jpg", "_reconstructed.jpg")
    os.makedirs(os.path.dirname(targert_path), exist_ok=True)
    cv2.imwrite(targert_path, reconstructed_image)
