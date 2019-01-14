import numpy as np


def sub_matrix(matrix, left_x, right_x, left_y, right_y):
    return matrix[np.ix_(range(left_x, right_x + 1), range(left_y, right_y + 1))]


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# reference: 10.3.3 Digital Image Processing
def compute_otsu_value(histogram):
    total_pixels = sum(histogram)
    num_value = len(histogram)

    # calculate p_i
    probability = histogram / total_pixels

    # calculate P1_k
    curr = 0
    cumulative_probability = list()
    for i in range(num_value):
        curr += probability[i]
        cumulative_probability.append(curr)

    # calculate m_G
    individual_mean = [intensity * count for intensity, count in enumerate(histogram)]
    global_mean = sum(individual_mean)

    # calculate m_k
    curr_mean = 0
    cumulative_mean = list()
    for i in range(num_value):
        curr_mean += individual_mean[i]
        cumulative_mean.append(curr_mean)

    # calculate between-class variance sigma_B
    sigma_B = list()
    for i in range(num_value):
        denominator = (cumulative_probability[i] * (1 - cumulative_probability[i]))
        if is_close(denominator, 0):
            sigma = 0
        else:
            sigma = pow(global_mean * cumulative_probability[i] - cumulative_mean[i], 2) / denominator
        sigma_B.append(sigma)

    # calculate max k
    max_sigma_value = np.amax(np.asarray(sigma_B))
    sigma_maxs = list()
    for i in range(num_value):
        if is_close(sigma_B[i], max_sigma_value):
            sigma_maxs.append(i)

    return int(sum(sigma_maxs) / len(sigma_maxs))


def get_angle_between_two_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    angle = np.arctan2([y1 - y2], [x1 - x2]) * 180 / np.pi
    return angle


def swap(x, y):
    return y, x


def check_2_line_segments_intersect(p1, q1, p2, q2):
    def on_segment(p, q, r):
        if min(p[0], p[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], p[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # colinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counter-clockwise

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return 1

    if o1 == 0 and on_segment(p1, p2, q1):
        return 1

    if o2 == 0 and on_segment(p1, q2, q1):
        return 1

    if o3 == 0 and on_segment(p2, p1, q2):
        return 1

    if o4 == 0 and on_segment(p2, q1, q2):
        return 1

    return 0


def check_point_inside_rect(point, rect_tl, rect_br):
    """
    Checking if a point is inside a rectangle.

    Note: Point (0, 0) is at top left corner.

    :param point:
    :param rect_tl: top left point of rectangle
    :param rect_br: bottom right point of rectangle
    :return:
    """
    if rect_tl[0] <= point[0] <= rect_br[0] and rect_tl[1] <= point[1] <= rect_br[1]:
        return 1
    return 0
