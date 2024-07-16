# %%
import cv2
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from recap import CfgNode as CN
import matplotlib.pyplot as plt

import typing
import multiprocessing
import warnings
import os

warnings.filterwarnings("ignore", message="The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
warnings.filterwarnings("ignore", message="found 0 physical cores < 1")

cpu_count = multiprocessing.cpu_count()
os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)  

cfg = {
    "RESIZE_IMAGE": {"WIDTH": 1200},
    "EDGE_DETECTION": {
        "LOW_THRESHOLD": 90,
        "HIGH_THRESHOLD": 400,
        "APERTURE": 3
    },
    "LINE_DETECTION": {
        "THRESHOLD": 150,
        "DIAGONAL_LINE_ELIMINATION": True,
        "DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES": 30
    },
    "BORDER_REFINEMENT": {
        "LINE_WIDTH": 4,
        "WARPED_SQUARE_SIZE": [50, 50],
        "NUM_SURROUNDING_SQUARES_IN_WARPED_IMG": 5,
        "SOBEL_KERNEL_SIZE": 3,
        "EDGE_DETECTION": {
            "HORIZONTAL": {
                "APERTURE": 3,
                "HIGH_THRESHOLD": 300,
                "LOW_THRESHOLD": 120
            },
            "VERTICAL": {
                "APERTURE": 3,
                "HIGH_THRESHOLD": 200,
                "LOW_THRESHOLD": 100
            }
        }
    },
    "MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE": 0.7,
    "RANSAC": {
        "BEST_SOLUTION_TOLERANCE": 0.15,
        "OFFSET_TOLERANCE": 0.1
    }
}


# %%
class ChessboardNotLocatedException(Exception):
    """Exception raised when the chessboard cannot be located in the image."""
    pass


# %%
def to_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Convert Cartesian to homogenous coordinates.

    Args:
        coordinates (np.ndarray): the Cartesian coordinates (shape: [..., 2])

    Returns:
        np.ndarray: the homogenous coordinates (shape: [..., 3])
    """
    return np.concatenate([coordinates,
                           np.ones((*coordinates.shape[:-1], 1))], axis=-1)


def from_homogenous_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Convert homogenous to Cartesian coordinates.

    Args:
        coordinates (np.ndarray): the homogenous coordinates (shape: [..., 3])

    Returns:
        np.ndarray: the Cartesian coordinates (shape: [..., 2])
    """
    return coordinates[..., :2] / coordinates[..., 2, np.newaxis]

# %%
def resize_image(cfg: dict, img: np.ndarray) -> np.ndarray:
    """Resize an image for use in the corner detection pipeline, maintaining the aspect ratio.

    Args:
        cfg (dict): the configuration dictionary
        img (np.ndarray): the input image

    Returns:
        np.ndarray: the resized image
    """
    h, w, _ = img.shape
    if w == cfg['RESIZE_IMAGE']['WIDTH']:
        return img, 1
    scale = cfg['RESIZE_IMAGE']['WIDTH'] / w
    dims = (cfg['RESIZE_IMAGE']['WIDTH'], int(h * scale))

    img = cv2.resize(img, dims)
    return img, scale


# %%
def _detect_edges(edge_detection_cfg: dict, gray: np.ndarray) -> np.ndarray:
    """Detect edges in a grayscale image.

    Args:
        edge_detection_cfg (dict): the edge detection configuration
        gray (np.ndarray): the grayscale image

    Returns:
        np.ndarray: the binary edge map
    """
    if gray.dtype != np.uint8:
        gray = gray / gray.max() * 255
        gray = gray.astype(np.uint8)
    edges = cv2.Canny(gray,
                      edge_detection_cfg['LOW_THRESHOLD'],
                      edge_detection_cfg['HIGH_THRESHOLD'],
                      edge_detection_cfg['APERTURE'])
    return edges


# %%
def _fix_negative_rho_in_hesse_normal_form_new(lines: np.ndarray) -> np.ndarray:
    """Fix negative rho values in Hesse normal form representation of lines.

    Args:
        lines (np.ndarray): array of lines in Hesse normal form

    Returns:
        np.ndarray: array of lines with positive rho values
    """
    lines = lines.copy()
    neg_rho_mask = lines[..., 0] < 0
    lines[neg_rho_mask, 0] = -lines[neg_rho_mask, 0]
    lines[neg_rho_mask, 1] -= np.pi
    return lines

# %%
def _detect_lines(cfg: dict, edges: np.ndarray) -> np.ndarray:
    """Detect lines in a binary edge map using the Hough transform.

    Args:
        cfg (dict): the configuration dictionary
        edges (np.ndarray): the binary edge map
    Returns:
        np.ndarray: an array of detected lines in Hesse normal form
    """
    # array of [rho, theta]
    lines = cv2.HoughLines(edges, 1, np.pi/360, cfg['LINE_DETECTION']['THRESHOLD'])
    lines = lines.squeeze(axis=-2)
    lines = _fix_negative_rho_in_hesse_normal_form_new(lines)

    if cfg['LINE_DETECTION']['DIAGONAL_LINE_ELIMINATION']:
        threshold = np.deg2rad(
            cfg['LINE_DETECTION']['DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES'])
        vmask = np.abs(lines[:, 1]) < threshold
        hmask = np.abs(lines[:, 1] - np.pi / 2) < threshold
        mask = vmask | hmask
        lines = lines[mask]
    return lines



# %%
def _fix_negative_rho_in_hesse_normal_form(lines: np.ndarray) -> np.ndarray:
    """
    Converts lines in Hesse normal form with negative rho values to standard positive rho.

    Parameters:
    lines (np.ndarray): Array of lines in Hesse normal form, where each line is defined by rho and theta.

    Returns:
    np.ndarray: Array of lines with all rho values non-negative and corresponding adjustments to theta.
    """
    lines = lines.copy()
    neg_rho_mask = lines[..., 0] < 0
    lines[neg_rho_mask, 0] = -lines[neg_rho_mask, 0]
    lines[neg_rho_mask, 1] = lines[neg_rho_mask, 1] - np.pi
    return lines

def _absolute_angle_difference(x, y):
    """
    Computes the absolute difference between two angles, considering the periodicity of angles.

    Parameters:
    x (float): First angle in radians.
    y (float): Second angle in radians.

    Returns:
    float: Minimal absolute difference between two angles in radians.
    """
    diff = np.mod(np.abs(x - y), 2*np.pi)
    return np.min(np.stack([diff, 2*np.pi - diff], axis=-1), axis=-1)

def _sort_lines(lines: np.ndarray) -> np.ndarray:
    """
    Sorts an array of lines by their rho values.

    Parameters:
    lines (np.ndarray): Array of lines in Hesse normal form.

    Returns:
    np.ndarray: Array of lines sorted by rho.
    """
    if lines.ndim == 0 or lines.shape[-2] == 0:
        return lines
    rhos = lines[..., 0]
    sorted_indices = np.argsort(rhos)
    return lines[sorted_indices]

def _cluster_horizontal_and_vertical_lines(lines: np.ndarray):
    """
    Clusters lines into horizontal and vertical groups based on their orientation.

    Parameters:
    lines (np.ndarray): Array of lines in Hesse normal form.

    Returns:
    tuple: Two np.ndarrays containing the horizontal and vertical lines respectively.
    """
    lines = _sort_lines(lines)
    thetas = lines[..., 1].reshape(-1, 1)
    distance_matrix = pairwise_distances(thetas, thetas, metric=_absolute_angle_difference)
    agg = AgglomerativeClustering(n_clusters=2, linkage="average")
    clusters = agg.fit_predict(distance_matrix)

    angle_with_y_axis = _absolute_angle_difference(thetas, 0.)
    if angle_with_y_axis[clusters == 0].mean() > angle_with_y_axis[clusters == 1].mean():
        hcluster, vcluster = 0, 1
    else:
        hcluster, vcluster = 1, 0

    horizontal_lines = lines[clusters == hcluster]
    vertical_lines = lines[clusters == vcluster]

    return horizontal_lines, vertical_lines



def _eliminate_similar_lines(lines: np.ndarray, perpendicular_lines: np.ndarray) -> np.ndarray:
    """
    Eliminates similar lines by clustering them based on their intersection points
    with a perpendicular line and retaining only the median line from each cluster.

    Parameters:
    lines (np.ndarray): Array of lines in Hesse normal form (rho, theta) that need to be filtered.
    perpendicular_lines (np.ndarray): Array of a single line in Hesse normal form that is 
    used as the reference perpendicular line for calculating intersection points.

    Returns:
    np.ndarray: Array of filtered lines, where only one line from each cluster of similar lines
                is retained based on their proximity to the perpendicular line.
    """
    perp_rho, perp_theta = perpendicular_lines.mean(axis=0)
    rho, theta = np.moveaxis(lines, -1, 0)
    intersection_points = get_intersection_point(
        rho, theta, perp_rho, perp_theta)
    intersection_points = np.stack(intersection_points, axis=-1)

    clustering = DBSCAN(eps=12, min_samples=1).fit(intersection_points)

    filtered_lines = []
    for c in range(clustering.labels_.max() + 1):
        lines_in_cluster = lines[clustering.labels_ == c]
        rho = lines_in_cluster[..., 0]
        median = np.argsort(rho)[len(rho)//2]
        filtered_lines.append(lines_in_cluster[median])
    return np.stack(filtered_lines)


def get_intersection_point(rho1: np.ndarray, theta1: np.ndarray, rho2: np.ndarray, theta2: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Obtain the intersection point of two lines in Hough space.

    This method can be batched

    Args:
        rho1 (np.ndarray): first line's rho
        theta1 (np.ndarray): first line's theta
        rho2 (np.ndarray): second lines's rho
        theta2 (np.ndarray): second line's theta

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: the x and y coordinates of the intersection point(s)
    """
    # rho1 = x cos(theta1) + y sin(theta1)
    # rho2 = x cos(theta2) + y sin(theta2)
    cos_t1 = np.cos(theta1)
    cos_t2 = np.cos(theta2)
    sin_t1 = np.sin(theta1)
    sin_t2 = np.sin(theta2)
    x = (sin_t1 * rho2 - sin_t2 * rho1) / (cos_t2 * sin_t1 - cos_t1 * sin_t2)
    y = (cos_t1 * rho2 - cos_t2 * rho1) / (sin_t2 * cos_t1 - sin_t1 * cos_t2)
    return x, y


def _choose_from_range(upper_bound: int, n: int = 2):
    """
    Randomly selects 'n' unique numbers from a range up to 'upper_bound'.

    Parameters:
    upper_bound (int): The upper bound of the range from which numbers are to be selected.
    n (int): The number of unique numbers to select.

    Returns:
    np.ndarray: An array of 'n' unique integers, sorted in ascending order, chosen from range(upper_bound).
    """
    return np.sort(np.random.choice(np.arange(upper_bound), (n,), replace=False), axis=-1)

def _get_intersection_points(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> np.ndarray:
    """
    Calculates intersection points between sets of horizontal and vertical lines.

    Parameters:
    horizontal_lines (np.ndarray): Array of horizontal lines in Hesse normal form (rho, theta).
    vertical_lines (np.ndarray): Array of vertical lines in Hesse normal form (rho, theta).

    Returns:
    np.ndarray: A 2D array where each row represents the x, y coordinates of the intersection point between
                a horizontal and a vertical line.
    """
    rho1, theta1 = np.moveaxis(horizontal_lines, -1, 0)
    rho2, theta2 = np.moveaxis(vertical_lines, -1, 0)

    rho1, rho2 = np.meshgrid(rho1, rho2, indexing="ij")
    theta1, theta2 = np.meshgrid(theta1, theta2, indexing="ij")
    intersection_points = get_intersection_point(rho1, theta1, rho2, theta2)
    intersection_points = np.stack(intersection_points, axis=-1)
    return intersection_points



def compute_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute the transformation matrix based on source and destination points.

    Args:
        src_points (np.ndarray): the source points (shape: [..., 2])
        dst_points (np.ndarray): the source points (shape: [..., 2])

    Returns:
        np.ndarray: the transformation matrix
    """
    transformation_matrix, _ = cv2.findHomography(src_points.reshape(-1, 2),
                                                  dst_points.reshape(-1, 2))
    return transformation_matrix


def _compute_homography(intersection_points: np.ndarray, row1: int, row2: int, col1: int, col2: int):
    """
    Computes the homography matrix from four selected intersection points.

    Parameters:
    intersection_points (np.ndarray): Array of intersection points where each point is in (x, y) format.
    row1, row2 (int): Indices of the rows of the points.
    col1, col2 (int): Indices of the columns of the points.

    Returns:
    np.ndarray: The computed homography matrix.
    """
    p1 = intersection_points[row1, col1]  # top left
    p2 = intersection_points[row1, col2]  # top right
    p3 = intersection_points[row2, col2]  # bottom right
    p4 = intersection_points[row2, col1]  # bottom left

    src_points = np.stack([p1, p2, p3, p4])
    dst_points = np.array([[0, 0],  # top left
                           [1, 0],  # top right
                           [1, 1],  # bottom right
                           [0, 1]])  # bottom left
    return compute_transformation_matrix(src_points, dst_points)


def _warp_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Warps points using the given transformation matrix.

    Parameters:
    transformation_matrix (np.ndarray): A homography or transformation matrix.
    points (np.ndarray): Array of points to be warped.

    Returns:
    np.ndarray: Array of warped points.
    """
    points = to_homogenous_coordinates(points)
    warped_points = points @ transformation_matrix.T
    return from_homogenous_coordinates(warped_points)

# %%
def _find_best_scale(cfg: dict, values: np.ndarray, scales: np.ndarray = np.arange(1, 9)):
    """
    Identifies the best scaling factor that minimizes the rounding error of scaled values.

    Parameters:
    cfg (dict): Configuration dictionary containing specific tolerances and parameters.
    values (np.ndarray): Array of values that need to be scaled.
    scales (np.ndarray): Array of potential scales to be applied.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the best scale and a mask that indicates which values are considered inliers under this scale.
    """
    scales = np.sort(scales)
    scaled_values = np.expand_dims(values, axis=-1) * scales
    diff = np.abs(np.rint(scaled_values) - scaled_values)

    inlier_mask = diff < cfg["RANSAC"]["OFFSET_TOLERANCE"] / scales
    num_inliers = np.sum(inlier_mask, axis=tuple(range(inlier_mask.ndim - 1)))

    best_num_inliers = np.max(num_inliers)

    # We will choose a slightly worse scale if it is lower
    index = np.argmax(num_inliers > (
        1 - cfg["RANSAC"]["BEST_SOLUTION_TOLERANCE"]) * best_num_inliers)
    return scales[index], inlier_mask[..., index]


def _discard_outliers(cfg: dict, warped_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float, float]:
    
    """
    Filters outliers from the detected points based on their adherence to expected grid scales.

    Parameters:
    cfg (dict): Configuration dictionary containing outlier detection parameters.
    warped_points (np.ndarray): Array of points in the warped image.
    intersection_points (np.ndarray): Array of intersection points from which the warped points were derived.

    Returns:
    Tuple[np.ndarray, np.ndarray, float, float]: A tuple containing filtered warped points, filtered intersection points, and the determined horizontal and vertical scales.
    """
    horizontal_scale, horizontal_mask = _find_best_scale(
        cfg, warped_points[..., 0])
    vertical_scale, vertical_mask = _find_best_scale(
        cfg, warped_points[..., 1])
    mask = horizontal_mask & vertical_mask

    # Keep rows/cols that have more than 50% inliers
    num_rows_to_consider = np.any(mask, axis=-1).sum()
    num_cols_to_consider = np.any(mask, axis=-2).sum()
    rows_to_keep = mask.sum(axis=-1) / num_rows_to_consider > \
        cfg["MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE"]
    cols_to_keep = mask.sum(axis=-2) / num_cols_to_consider > \
        cfg["MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE"]

    warped_points = warped_points[rows_to_keep][:, cols_to_keep]
    intersection_points = intersection_points[rows_to_keep][:, cols_to_keep]
    return warped_points, intersection_points, horizontal_scale, vertical_scale


def _quantize_points(cfg: dict, warped_scaled_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Quantizes points to a grid and normalizes their positions according to a defined scale and configuration.

    Parameters:
    cfg (dict): Configuration dictionary with parameters for border refinement and scaling.
    warped_scaled_points (np.ndarray): Array containing the scaled coordinates of points in the warped image.
    intersection_points (np.ndarray): Array of intersection points derived from line detection.

    Returns:
    Tuple[tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        - A tuple containing the boundaries of the grid (xmin, xmax, ymin, ymax).
        - The scale used for final positioning of the grid.
        - The scaled and quantized positions of the grid points.
        - The filtered intersection points after removing outliers.
        - The dimensions of the warped image after quantization and scaling.

    This function quantizes the positions of intersection points to a regular grid, removes duplicate points,
    and adjusts the grid to a predefined number of rows and columns. It ensures that the points are
    aligned to a grid in the warped space, allowing for more accurate subsequent analyses or transformations.
    """
    mean_col_xs = warped_scaled_points[..., 0].mean(axis=0)
    mean_row_ys = warped_scaled_points[..., 1].mean(axis=1)

    col_xs = np.rint(mean_col_xs).astype(np.int32)
    row_ys = np.rint(mean_row_ys).astype(np.int32)

    # Remove duplicates
    col_xs, col_indices = np.unique(col_xs, return_index=True)
    row_ys, row_indices = np.unique(row_ys, return_index=True)
    intersection_points = intersection_points[row_indices][:, col_indices]

    # Compute mins and maxs in warped space
    xmin = col_xs.min()
    xmax = col_xs.max()
    ymin = row_ys.min()
    ymax = row_ys.max()

    # Ensure we a have a maximum of 9 rows/cols
    while xmax - xmin > 9:
        xmax -= 1
        xmin += 1
    while ymax - ymin > 9:
        ymax -= 1
        ymin += 1
    col_mask = (col_xs >= xmin) & (col_xs <= xmax)
    row_mask = (row_ys >= xmin) & (row_ys <= xmax)

    # Discard
    col_xs = col_xs[col_mask]
    row_ys = row_ys[row_mask]
    intersection_points = intersection_points[row_mask][:, col_mask]

    # Create quantized points array
    quantized_points = np.stack(np.meshgrid(col_xs, row_ys), axis=-1)

    # Transform in warped space
    translation = -np.array([xmin, ymin]) + \
        cfg["BORDER_REFINEMENT"]["NUM_SURROUNDING_SQUARES_IN_WARPED_IMG"]
    scale = np.array(cfg["BORDER_REFINEMENT"]["WARPED_SQUARE_SIZE"])

    scaled_quantized_points = (quantized_points + translation) * scale
    xmin, ymin = np.array((xmin, ymin)) + translation
    xmax, ymax = np.array((xmax, ymax)) + translation
    warped_img_size = (np.array((xmax, ymax)) +
                       cfg["BORDER_REFINEMENT"]["NUM_SURROUNDING_SQUARES_IN_WARPED_IMG"]) * scale

    return (xmin, xmax, ymin, ymax), scale, scaled_quantized_points, intersection_points, warped_img_size


def _compute_vertical_borders(cfg: dict, warped: np.ndarray, mask: np.ndarray, scale: np.ndarray, xmin: int, xmax: int) -> typing.Tuple[int, int]:
    """
    Computes the optimal vertical borders of an image by detecting strong edges after applying Sobel filtering.

    Parameters:
    cfg (dict): Configuration dictionary containing settings for edge detection and border refinement.
    warped (np.ndarray): Warped image array where vertical borders are to be determined.
    mask (np.ndarray): A boolean array that masks the valid area of interest in the image.
    scale (np.ndarray): Scaling factors applied to the image dimensions.
    xmin (int): Initial minimum x-coordinate for the vertical border.
    xmax (int): Initial maximum x-coordinate for the vertical border.

    Returns:
    Tuple[int, int]: A tuple containing the updated minimum and maximum x-coordinates after border refinement.

    The function iteratively adjusts the xmin and xmax coordinates by evaluating the sum of edge intensities
    on potential new border positions, ensuring that the strongest edges define the vertical boundaries of the
    region of interest.
    """
    G_x = np.abs(cv2.Sobel(warped, cv2.CV_64F, 1, 0,
                           ksize=cfg["BORDER_REFINEMENT"]["SOBEL_KERNEL_SIZE"]))
    G_x[~mask] = 0
    G_x = _detect_edges(cfg["BORDER_REFINEMENT"]["EDGE_DETECTION"]["VERTICAL"], G_x)
    G_x[~mask] = 0

    def get_nonmax_supressed(x):
        x = (x * scale[0]).astype(np.int32)
        thresh = cfg["BORDER_REFINEMENT"]["LINE_WIDTH"] // 2
        return G_x[:, x-thresh:x+thresh+1].max(axis=1)

    while xmax - xmin < 8:
        top = get_nonmax_supressed(xmax + 1)
        bottom = get_nonmax_supressed(xmin - 1)

        if top.sum() > bottom.sum():
            xmax += 1
        else:
            xmin -= 1

    return xmin, xmax


def _compute_horizontal_borders(cfg: dict, warped: np.ndarray, mask: np.ndarray, scale: np.ndarray, ymin: int, ymax: int) -> typing.Tuple[int, int]:
    """
    Computes the optimal horizontal borders of an image by detecting strong edges after applying Sobel filtering.

    Parameters:
    cfg (dict): Configuration dictionary containing settings for edge detection and border refinement.
    warped (np.ndarray): Warped image array where horizontal borders are to be determined.
    mask (np.ndarray): A boolean array that masks the valid area of interest in the image.
    scale (np.ndarray): Scaling factors applied to the image dimensions.
    ymin (int): Initial minimum y-coordinate for the horizontal border.
    ymax (int): Initial maximum y-coordinate for the horizontal border.

    Returns:
    Tuple[int, int]: A tuple containing the updated minimum and maximum y-coordinates after border refinement.

    This function similarly adjusts the ymin and ymax coordinates based on the intensity of detected edges,
    refining the border to tightly fit the primary subject of the image while avoiding unnecessary background.
    """
    G_y = np.abs(cv2.Sobel(warped, cv2.CV_64F, 0, 1,
                           ksize=cfg["BORDER_REFINEMENT"]["SOBEL_KERNEL_SIZE"]))
    G_y[~mask] = 0
    G_y = _detect_edges(cfg["BORDER_REFINEMENT"]["EDGE_DETECTION"]["HORIZONTAL"], G_y)
    G_y[~mask] = 0

    def get_nonmax_supressed(y):
        y = (y * scale[1]).astype(np.int32)
        thresh = cfg["BORDER_REFINEMENT"]["LINE_WIDTH"] // 2
        return G_y[y-thresh:y+thresh+1].max(axis=0)

    while ymax - ymin < 8:
        top = get_nonmax_supressed(ymax + 1)
        bottom = get_nonmax_supressed(ymin - 1)

        if top.sum() > bottom.sum():
            ymax += 1
        else:
            ymin -= 1
    return ymin, ymax


# %%
def find_corners(cfg: dict, img: str) -> np.ndarray:
    """Determine the four corner points of the chessboard in an image.

    Args:
        cfg (dict): the configuration dictionary
        img_path (str): the path to the input image

    Returns:
        np.ndarray: the pixel coordinates of the four corners
    """
    # img = cv2.imread(img_path)
    img, img_scale = resize_image(cfg, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = _detect_edges(cfg['EDGE_DETECTION'], gray)
    lines = _detect_lines(cfg, edges)
    if lines.shape[0] > 400:
        raise ChessboardNotLocatedException("too many lines in the image")
    all_horizontal_lines, all_vertical_lines = _cluster_horizontal_and_vertical_lines(
        lines)

    horizontal_lines = _eliminate_similar_lines(
        all_horizontal_lines, all_vertical_lines)
    vertical_lines = _eliminate_similar_lines(
        all_vertical_lines, all_horizontal_lines)

    all_intersection_points = _get_intersection_points(horizontal_lines,
                                                           vertical_lines)

    best_num_inliers = 0
    best_configuration = None
    iterations = 0
    while iterations < 200 or best_num_inliers < 30:
        row1, row2 = _choose_from_range(len(horizontal_lines))
        col1, col2 = _choose_from_range(len(vertical_lines))
        transformation_matrix = _compute_homography(all_intersection_points,
                                                        row1, row2, col1, col2)
        warped_points = _warp_points(
            transformation_matrix, all_intersection_points)
        warped_points, intersection_points, horizontal_scale, vertical_scale = _discard_outliers(
            cfg, warped_points, all_intersection_points)
        num_inliers = np.prod(warped_points.shape[:-1])
        if num_inliers > best_num_inliers:
            warped_points *= np.array((horizontal_scale, vertical_scale))

            # Quantize and reject deuplicates
            (xmin, xmax, ymin, ymax), scale, quantized_points, intersection_points, warped_img_size = configuration = _quantize_points(
                cfg, warped_points, intersection_points)

            # Calculate remaining number of inliers
            num_inliers = np.prod(quantized_points.shape[:-1])

            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_configuration = configuration
        iterations += 1
        if iterations > 10000:
            raise ChessboardNotLocatedException(
                "RANSAC produced no viable results")

    # Retrieve best configuration
    (xmin, xmax, ymin, ymax), scale, quantized_points, intersection_points, warped_img_size = best_configuration

    # Recompute transformation matrix based on all inliers
    transformation_matrix = compute_transformation_matrix(
        intersection_points, quantized_points)
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Warp grayscale image
    dims = tuple(warped_img_size.astype(np.int32))
    warped = cv2.warpPerspective(gray, transformation_matrix, dims)
    borders = np.zeros_like(gray)
    borders[3:-3, 3:-3] = 1
    warped_borders = cv2.warpPerspective(borders, transformation_matrix, dims)
    warped_mask = warped_borders == 1

    # Refine board boundaries
    xmin, xmax = _compute_vertical_borders(
        cfg, warped, warped_mask, scale, xmin, xmax)
    scaled_xmin, scaled_xmax = (int(x * scale[0]) for x in (xmin, xmax))
    warped_mask[:, :scaled_xmin] = warped_mask[:, scaled_xmax:] = False
    ymin, ymax = _compute_horizontal_borders(
        cfg, warped, warped_mask, scale, ymin, ymax)

    # Transform boundaries to image space
    corners = np.array([[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]]).astype(np.float32)
    corners = corners * scale
    img_corners = _warp_points(inverse_transformation_matrix, corners)
    return sort_corner_points(img_corners), vertical_lines, horizontal_lines

def sort_corner_points(points: np.ndarray) -> np.ndarray:
    """Permute the board corner coordinates to the order [top left, top right, bottom right, bottom left].

    Args:
        points (np.ndarray): the four corner coordinates

    Returns:
        np.ndarray: the permuted array
    """

    # First, order by y-coordinate
    points = points[points[:, 1].argsort()]
    # Sort top x-coordinates
    points[:2] = points[:2][points[:2, 0].argsort()]
    # Sort bottom x-coordinates (reversed)
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]

    return points

# %%
def draw_lines(img, lines):
    """
    Draws lines on an image based on their Hesse normal form parameters.

    Parameters:
    img (np.ndarray): The image array on which lines will be drawn. This image should be in the format
                      acceptable by OpenCV, typically (height, width, channels).
    lines (iterable of tuples): An iterable of tuples, each containing two elements (rho, theta) representing
                                a line in Hesse normal form. Here, `rho` is the distance from the origin to
                                the line, and `theta` is the angle in radians between the x-axis and the line.

    Effects:
    This function modifies the input image in-place by drawing each line specified in the `lines` parameter.
    Each line is drawn using the BGR color [0, 0, 255] (red) and a thickness of 2 pixels.

    Example usage:
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    lines = [(100, np.pi/4), (200, np.pi/3)]
    draw_lines(img, lines)
    cv2.imshow('Lines on Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    Notes:
    This function calculates the endpoints of each line by converting the polar coordinates (rho, theta) into
    Cartesian coordinates and extending them significantly in both directions (1000 pixels) from the line's
    midpoint. This ensures that lines extend across the entire image, regardless of the image's size.
    """
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


# %%
def is_segment_intersecting_box(x1, y1, x2, y2, corners, shift_corners):
    """
    Determines if a line segment intersects any of the four sides of a quadrilateral.

    Parameters:
    x1, y1 (float): The x and y coordinates of the first point of the line segment.
    x2, y2 (float): The x and y coordinates of the second point of the line segment.
    corners (array-like of tuples): List of tuples containing the coordinates of the quadrilateral's corners.
    shift_corners (array-like of tuples): List of tuples containing the coordinates of the shifted quadrilateral's corners.

    Returns:
    bool: True if the line segment intersects any side of the quadrilateral, False otherwise.

    Description:
    This function checks intersection between a specified line segment and all sides of a quadrilateral.
    The corners of the quadrilateral are supposed to be connected in a sequence, and shift_corners are
    provided to potentially handle a transformed or shifted version of the original quadrilateral.
    """
    for i in range(4):
        x3, y3 = shift_corners[i]
        x4, y4 = shift_corners[(i + 1) % 4]

        if is_segments_intersecting(x1, y1, x2, y2, x3, y3, x4, y4):
            return True
    return False

def is_segments_intersecting(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Determines if two line segments intersect.

    Parameters:
    x1, y1 (float): The x and y coordinates of the first endpoint of the first line segment.
    x2, y2 (float): The x and y coordinates of the second endpoint of the first line segment.
    x3, y3 (float): The x and y coordinates of the first endpoint of the second line segment.
    x4, y4 (float): The x and y coordinates of the second endpoint of the second line segment.

    Returns:
    bool: True if the segments intersect, False otherwise.

    Description:
    The function determines the intersection by calculating the cross products of the vectors formed by the
    endpoints of the segments. The segments intersect if the endpoints of one segment lie on opposite sides
    of the other segment.
    """
    vec1 = (x2 - x1, y2 - y1)
    vec2 = (x3 - x1, y3 - y1)
    vec3 = (x4 - x1, y4 - y1)

    cross_product1 = np.cross(vec1, vec2)
    cross_product2 = np.cross(vec1, vec3)

    if cross_product1 * cross_product2 < 0:
        return True
    else:
        return False



# %%
def shift_square_points(square_points, shift_amount = 5):
    """
    Shifts each of the four points a couple of pixels toward the center of the square.
    
    Arguments:
        square_points (list): A list of the points that form the square, in (x, y) format.
    
    Returns:
        list: A list of shifted points.
    """
    center_x = sum(point[0] for point in square_points) / 4
    center_y = sum(point[1] for point in square_points) / 4

    shifted_points = []
    for point in square_points:
        shifted_x = point[0] + (center_x - point[0]) / abs(center_x - point[0]) * shift_amount
        shifted_y = point[1] + (center_y - point[1]) / abs(center_y - point[1]) * shift_amount
        shifted_points.append((shifted_x, shifted_y))

    return shifted_points


# %%

def draw_points(image, points, color='g', marker='o', markersize=5):
     
    """
    Draws points on an image using the Matplotlib library.
    
    Arguments:
        image (numpy.ndarray): The image on which to draw points.
        points (list): A list of points in the format (x, y).
        color (str): The color of the points.
        marker (str): The character used to mark the points.
        markersize (int): The size of the dots.
    
    Returns:
        None
    """
    plt.figure()

    plt.imshow(image)

    for point in points:
        plt.plot(point[0], point[1], marker, color=color, markersize=markersize)

    plt.show()


# %%
def transform_perspective(image, corners, lines):
    """
    Applies perspective transformation to the image and lines.
    
    Args:
        image (numpy.ndarray): Source image.
        corners (list): A list of the coordinates of the four corners of the source area.
        lines (list): List of coordinates of the start and end points of the lines before the transformation.
    
    Returns:
        tuple: A tuple containing the transformed image and the transformed line coordinates.
    """
    padding = 6  
    padding_pixels = padding * 20  

    padding_corners = np.array([
        [corners[0, 0] - padding_pixels, corners[0, 1] - padding_pixels],
        [corners[1, 0] + padding_pixels, corners[1, 1] - padding_pixels],
        [corners[2, 0] + padding_pixels, corners[2, 1] + padding_pixels],
        [corners[3, 0] - padding_pixels, corners[3, 1] + padding_pixels]
    ], dtype=np.float32)

    dst_corners = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(padding_corners, dst_corners)

    warped_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    new_corners = cv2.perspectiveTransform(np.array([corners], dtype=np.float32), matrix)[0]

    lines = [[[line[0], line[1]], [line[2], line[3]]] for line in lines]

    new_lines = cv2.perspectiveTransform(np.array(lines, dtype=np.float32), matrix)

    new_lines = [tuple(line.squeeze().astype(int).tolist()) for line in new_lines]

    return warped_image, new_corners, new_lines


# %%
def find_intersection(line1, line2):
    """
    Finds the intersection point of two lines given by their endpoints.

    Parameters:
        line1 (tuple): Tuple containing the coordinates of the endpoints of the first line segment ((x1, y1), (x2, y2)).
        line2 (tuple): Tuple containing the coordinates of the endpoints of the second line segment ((x3, y3), (x4, y4)).

    Returns:
        tuple or None: The coordinates of the intersection point as a tuple (x, y). Returns None if the lines are parallel.

    Description:
    This function calculates the intersection point of two lines specified by their endpoints. It uses the formula
    for the intersection of two lines in the form Ax + By = C, where A, B, and C are coefficients derived from
    the endpoints of the lines. If the lines are parallel (i.e., their determinant is zero), the function returns None.
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    det = A1 * B2 - A2 * B1

    if det == 0:
        return None

    x = int((B2 * C1 - B1 * C2) / det)
    y = int((A1 * C2 - A2 * C1) / det)

    return x, y



# %%
def crop_square_with_padding(image, top_left, bottom_right, padding_left=10, padding_right=10, padding_top=30, padding_bottom=0):
    """
    Crops a square region from an image, adding specified padding around the cropping area.

    Parameters:
        image (ndarray): The input image from which a square will be cropped.
        top_left (tuple): The coordinates (x, y) of the top-left corner of the cropping area.
        bottom_right (tuple): The coordinates (x, y) of the bottom-right corner of the cropping area.
        padding_left (int, optional): The amount of padding to add on the left side. Default is 10.
        padding_right (int, optional): The amount of padding to add on the right side. Default is 10.
        padding_top (int, optional): The amount of padding to add on the top side. Default is 30.
        padding_bottom (int, optional): The amount of padding to add on the bottom side. Default is 0.

    Returns:
        ndarray: The cropped square region from the image.

    Description:
    This function crops a square region from an image based on specified top-left and bottom-right coordinates,
    incorporating additional padding on each side as specified. The function first adjusts the coordinates based
    on the padding values, then calculates the size of the square to ensure it remains within the bounds of the
    original image dimensions. The cropped square is then extracted and returned.
    """
    top_x, top_y = top_left
    bottom_x, bottom_y = bottom_right

    top_x -= padding_left
    top_y -= padding_top
    bottom_x += padding_right
    bottom_y += padding_bottom

    size = min(bottom_x - top_x, bottom_y - top_y)

    cropped_image = image[top_y:top_y + size, top_x:top_x + size]

    return cropped_image


# %%

chessboard_list1 = []
for row in range(8, 0, -1):
    for col in range(ord('a'), ord('h') + 1):
        chessboard_list1.append(chr(col) + str(row))

# Создание списка клеток начиная с H1
chessboard_list2 = []
for row in range(1, 9):
    for col in range(ord('h'), ord('a') - 1, -1):
        chessboard_list2.append(chr(col) + str(row))



# %%
import json

def read_chess_position(json_file):
    """
    Reads a chess position from a JSON file containing FEN notation and returns a dictionary representing the board.

    Parameters:
    json_file (str): The path to the JSON file containing the chess position in FEN notation.

    Returns:
    dict: A dictionary with keys as chessboard squares (e.g., 'a1', 'h8') and values as pieces ('K', 'Q', 'p', etc.)
          or 'empty' if the square is unoccupied.

    Description:
    The function opens the specified JSON file, reads the FEN string, and parses it to construct a dictionary that
    represents the current state of the chessboard. Each square of the chessboard is identified by a coordinate such
    as 'a1' or 'h8', and the corresponding value is either a chess piece or 'empty' if no piece occupies that square.
    The dictionary covers all 64 squares of the board, ensuring each position is accounted for, regardless of whether
    it is explicitly mentioned in the FEN string.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    fen = data['fen']
    rows = fen.split('/')

    chessboard = {}

    for row_index, row in enumerate(rows):
        column_index = 0
        for symbol in row:
            if symbol.isdigit():
                column_index += int(symbol)
            else:
                column = chr(ord('a') + column_index)
                row_number = 8 - row_index
                square = f"{column}{row_number}"
                chessboard[square] = symbol.lower() if symbol.islower() else symbol.upper()
                column_index += 1
        for empty_column in range(column_index, 8):
            column = chr(ord('a') + empty_column)
            row_number = 8 - row_index
            square = f"{column}{row_number}"
            chessboard[square] = "empty"

    for row_number in range(8, 0, -1):
        for column in range(ord('a'), ord('h')+1):
            square = f"{chr(column)}{row_number}"
            if square not in chessboard:
                chessboard[square] = "empty"

    return chessboard




# %%

def create_chessboard_list():
    
    """
    Creates two lists representing the coordinates of a standard chessboard in different traversal orders.

    Returns:
    tuple of list: Two lists of string coordinates for the chessboard squares.
        - The first list (chessboard_list1) is ordered from 'a8' to 'h1' (top-left to bottom-right).
        - The second list (chessboard_list2) is ordered from 'h1' to 'a8' (bottom-right to top-left).

    Description:
    The function generates two lists of chessboard coordinates. The first list starts at 'a8' and moves horizontally 
    to 'h8', then down to the next row until 'h1'. The second list starts at 'h1' and moves horizontally to 'a1', then 
    up to the next row until 'a8'. These lists are useful for operations that need to traverse the chessboard in 
    specific orders, such as generating visual representations or initializing game states.
    """
    
    chessboard_list1 = []
    for row in range(8, 0, -1):
        for col in range(ord('a'), ord('h') + 1):
            chessboard_list1.append(chr(col) + str(row))

    chessboard_list2 = []
    for row in range(1, 9):
        for col in range(ord('h'), ord('a') - 1, -1):
            chessboard_list2.append(chr(col) + str(row))

    return chessboard_list1, chessboard_list2

# %%

def process_chessboard(img_link, json_link, output_folder):
    
    """
    Process a chessboard image to identify and extract each square, save them as images,
    and store their positions based on the chessboard configuration provided in a JSON file.

    Args:
        img_link (str): Path to the image file containing the chessboard.
        json_link (str): Path to the JSON file containing configuration, such as FEN string for piece placement.
        output_folder (str): Directory path where the processed images and JSON files will be saved.

    Process:
    - Read the image and JSON file.
    - Find corners of the chessboard in the image.
    - Check lines for intersection with a slightly shifted version of the corners to determine valid lines.
    - Transform the perspective of the image to align the chessboard for easier processing.
    - Filter and sort detected lines into vertical and horizontal lines.
    - Based on the sorted lines, identify intersections that define the boundaries of each chess square.
    - Extract and save each chess square as an image and its corresponding position in a JSON file in the output directory.

    The output directory will contain images named after their chessboard coordinates (e.g., 'a1.jpg', 'b1.jpg', etc.)
    and JSON files with chess piece information for each coordinate.

    Errors during processing are printed directly to console with indication of the problematic file.
    """
    
    counter = 0
    img = cv2.imread(img_link)
    corners, line1, line2 = find_corners(cfg, img)

    shift_corners = shift_square_points(corners, 10)
    
    valid_lines1 = []
    valid_lines2 = []

    for rho, theta in line1:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if is_segment_intersecting_box(x1, y1, x2, y2, corners, shift_corners):
            valid_lines1.append((x1, y1, x2, y2))

    for rho, theta in line2:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if is_segment_intersecting_box(x1, y1, x2, y2, corners, shift_corners):
            valid_lines2.append((x1, y1, x2, y2))

    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        if (x1, y1, x2, y2) not in valid_lines1:
            valid_lines1.append((x1, y1, x2, y2))

    lines = valid_lines1 + valid_lines2
    
    warped_image, new_corners, new_lines = transform_perspective(img, corners, lines)

    shift_corners = shift_square_points(new_corners, 10)

    # Предположим, что new_lines - это список всех линий
    vertical_lines = []
    horizontal_lines = []

    # Фильтрация списка линий на вертикальные и горизонтальные
    for line in new_lines:
        if abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]):
            vertical_lines.append(line)
        else:
            horizontal_lines.append(line)

    sorted_vertical_lines = sorted(vertical_lines, key=lambda line: (line[0][0]))


    sorted_horizontal_lines = sorted(horizontal_lines, key=lambda line: (line[0][1]))

    chessboard_list1, chessboard_list2  = create_chessboard_list()

    num_rows = 8  # Предполагая, что верхняя и нижняя границы не считаются линиями
    num_cols = 8  # Предполагая, что левая и правая границы не считаются линиями

    data = json.load(open(json_link, 'rb'))
    image = warped_image.copy()

    if data['white_turn'] == False:
        chessboard_list = chessboard_list2
    else:
        chessboard_list = chessboard_list1

    # Предположим, что у вас есть списки вертикальных и горизонтальных линий
    cnt = 0
    for row in range(num_rows):
        for col in range(num_cols):
            # Вычислите координаты текущей клетки
            intersection_point1 = find_intersection(sorted_horizontal_lines[row], sorted_vertical_lines[col])
            intersection_point2 = find_intersection(sorted_horizontal_lines[row + 1], sorted_vertical_lines[col + 1])
            # Вырезание текущей клетки из исходного изображения
            if row == 0:
                result = crop_square_with_padding(image, intersection_point1, intersection_point2, padding_top=80)
            elif row == 1:
                result = crop_square_with_padding(image, intersection_point1, intersection_point2, padding_top=60)
            elif row == 2:
                result = crop_square_with_padding(image, intersection_point1, intersection_point2, padding_top=40)
            else:
                result = crop_square_with_padding(image, intersection_point1, intersection_point2)


            os.makedirs(output_folder, exist_ok=True)
            chessboard = read_chess_position(json_link)
            # Сохраняем изображение в папку
            image_path = os.path.join(output_folder, f"{chessboard_list[cnt]}.jpg")
            try:
                cv2.imwrite(image_path, result)
                 # Сохраняем JSON файл в папку
                json_data = {"value": chessboard.get(chessboard_list[cnt], 'empty')}
                json_path = os.path.join(output_folder, f"{chessboard_list[cnt]}.json")
                with open(json_path, "w") as json_file:
                    json.dump(json_data, json_file)
                counter += 1
                
            except:
                print(f"Missing value at file {img_link}, {image_path}")

            cnt += 1


    print(f"Img No: {img_link}. Processing complete. cnt = {counter}")



def find_png_files(image_folder):
    """
    Find all PNG files within a specified directory.

    Args:
        image_folder (str): The path to the directory where the PNG files are located.

    Returns:
        list: A list of filenames that end with '.png' in the specified directory.

    This function checks if the specified directory exists, lists all files in the directory,
    and filters out those that have a '.png' extension. If the directory does not exist,
    it prints a warning message and returns an empty list.
    """
    if os.path.exists(image_folder):
        files = os.listdir(image_folder)
        png_files = [file for file in files if file.lower().endswith(".png")]
    else:
        print("Image folder not found.")

    return png_files

def create_variables_for_png_files(image_folder):
    """
    Process each PNG file in the specified directory by creating necessary variables and processing them.

    Args:
        image_folder (str): The path to the directory containing PNG files.

    This function finds all PNG files in the specified directory, generates necessary links for each file,
    and attempts to process them using the process_chessboard function. It handles exceptions by printing an error message.
    """
    png_files = find_png_files(image_folder)
    
    for png_file in png_files:
        img_name = os.path.splitext(os.path.basename(png_file))[0]
        img_link = f"{image_folder}/{img_name}.png"
        json_link = f"{image_folder}/{img_name}.json"
        output_folder = f"result/{img_name}"
        
        print(output_folder)
        
        try:
            process_chessboard(img_link, json_link, output_folder)
        except:
            print("Something's wrong.")



# create_variables_for_png_files("app_data")


