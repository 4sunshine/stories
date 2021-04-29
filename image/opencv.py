import cv2
import numpy as np


def two_parabolas_across_dimension_np(dim_size, maxima):
    points = np.arange(float(dim_size))
    left_side = points[:maxima] / maxima
    ls = -left_side * left_side + 2. * left_side
    right_side = points[maxima:]
    rs = (-right_side*right_side + 2 * maxima * right_side + (dim_size - 1) ** 2 - 2 * maxima * (dim_size - 1)) /\
         (dim_size - 1 - maxima)**2
    return np.concatenate([ls, rs]).astype(np.float32)


def prepare_grid_np(h, w):
    ys = np.arange(float(h)).astype(np.float32)
    xs = np.arange(float(w)).astype(np.float32)
    return np.meshgrid(xs, ys), xs, ys


def rainbow_image(h, w, horizontal=True):
    RAINBOW_START = 0
    RAINBOW_END = 150
    if horizontal:
        space = np.linspace(RAINBOW_START, RAINBOW_END, w, dtype=np.uint8)[np.newaxis, :]
    else:
        space = np.linspace(RAINBOW_START, RAINBOW_END, h, dtype=np.uint8)[:, np.newaxis]
    hue = np.ones((h, w), dtype=np.uint8)
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    hue *= space
    hsv = cv2.merge((hue, 255 * sat, 255 * val))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def grid_mask(h, w, n_y, n_x):
    h_space = np.linspace(0, h, n_y + 2, dtype=np.int32)[1: -1]
    w_space = np.linspace(0, w, n_x + 2, dtype=np.int32)[1: -1]
    h_start_coords = np.stack([np.zeros_like(h_space), h_space], axis=1)
    h_end_coords = np.stack([(w - 1) * np.ones_like(h_space), h_space], axis=1)
    w_start_coords = np.stack([w_space, np.zeros_like(w_space)], axis=1)
    w_end_coords = np.stack([w_space, (h - 1) * np.ones_like(w_space)], axis=1)
    mask = np.zeros((h, w), dtype=np.uint8)
    for h_s, h_e in zip(h_start_coords, h_end_coords):
        x_s, y_s = h_s
        x_e, y_e = h_e
        cv2.line(mask, (x_s, y_s), (x_e, y_e), 1, 2)
        cv2.putText(mask, str(int(y_s)), (x_s, y_s), cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)

    for w_s, w_e in zip(w_start_coords, w_end_coords):
        x_s, y_s = w_s
        x_e, y_e = w_e
        cv2.line(mask, (x_s, y_s), (x_e, y_e), 1, 2)
        cv2.putText(mask, str(int(x_s)), (x_s, y_s + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
    return mask


def rainbow_grid(h=720, w=1280, n_y=6, n_x=11, horizontal=True):
    image = rainbow_image(h, w, horizontal)
    mask = grid_mask(h, w, n_y, n_x)
    bgr_mask = cv2.merge((mask, mask, mask))
    return image * bgr_mask, bgr_mask, image


# NEED TO REMOVE THESE VARIABLES LATER
r_image, r_mask, r_full = rainbow_grid()
(x_grid, y_grid), xs, ys = prepare_grid_np(720, 1280)
x_grid = x_grid.astype(np.float32)
y_grid = y_grid.astype(np.float32)


def binded_warp_cv_displacements(image, bind_point, binded_dx):
    h, w, c = image.shape
    ones = np.ones((h, w), dtype=np.float32)
    binded_dx = np.array(binded_dx).astype(np.int32)
    dx = ones * binded_dx[0]
    dy = ones * binded_dx[1]
    y_coeff = two_parabolas_across_dimension_np(h, bind_point[1])
    x_coeff = two_parabolas_across_dimension_np(w, bind_point[0])

    if binded_dx[0] != 0:
        dxs = x_coeff * binded_dx[0]
        new_xs = xs + dxs
        new_dxs = np.interp(xs, new_xs, dxs)
        x_coeff = new_dxs / binded_dx[0]

    if binded_dx[1] != 0:
        dys = y_coeff * binded_dx[1]
        new_ys = ys + dys
        new_dys = np.interp(ys, new_ys, dys)
        y_coeff = new_dys / binded_dx[1]

    y_coeff = y_coeff[:, np.newaxis].astype(np.float32)
    x_coeff = x_coeff[np.newaxis, :].astype(np.float32)

    x_coords = xs
    graph = np.stack([x_coords, h - (h / 4.) * x_coeff[0]], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (255, 255, 0), 2)

    y_coords = ys
    graph = np.stack([w - (w / 7.) * y_coeff[:, 0], y_coords], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (0, 255, 252), 2)
    cv2.line(image, (bind_point[0] + binded_dx[0], h - 1), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (255, 255, 0))
    cv2.line(image, (w-1, bind_point[1] + binded_dx[1]), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (0, 255, 252))
    cv2.circle(image, (bind_point[0], bind_point[1]), 8, (255, 255, 255), -1)
    cv2.circle(image, (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), 6, (0, 102, 255), -1)

    xy_coeff = x_coeff * y_coeff

    hue = 127. * xy_coeff
    hue = hue.astype(np.uint8)
    sat = 255 * np.ones((h, w), dtype=np.uint8)
    bri = 255 * np.ones((h, w), dtype=np.uint8)
    hsv = cv2.merge((hue, sat, bri))
    power = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    dx_ = dx * x_coeff * y_coeff
    dy_ = dy * x_coeff * y_coeff

    x_grid_ = x_grid - dx_
    y_grid_ = y_grid - dy_

    image = cv2.remap(image, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    for i in range(0, h, 40):
        for j in range(0, w, 40):
            cv2.line(image, (int(x_grid[i, j]), int(y_grid[i, j])), (int(x_grid_[i, j]), int(y_grid_[i, j])),
                     tuple(power[i, j].tolist()), 1)
            cv2.circle(image, (int(x_grid[i, j]), int(y_grid[i, j])), 1,
                     tuple(power[i, j].tolist()), -1)

    return image


def binded_warp_cv_coloured(image, bind_point, binded_dx):
    h, w, c = image.shape
    ones = np.ones((h, w), dtype=np.float32)
    binded_dx = np.array(binded_dx).astype(np.int32)
    dx = ones * binded_dx[0]
    dy = ones * binded_dx[1]
    y_coeff = two_parabolas_across_dimension_np(h, bind_point[1])
    x_coeff = two_parabolas_across_dimension_np(w, bind_point[0])

    if binded_dx[0] != 0:
        dxs = x_coeff * binded_dx[0]
        new_xs = xs + dxs
        new_dxs = np.interp(xs, new_xs, dxs)
        x_coeff = new_dxs / binded_dx[0]

    if binded_dx[1] != 0:
        dys = y_coeff * binded_dx[1]
        new_ys = ys + dys
        new_dys = np.interp(ys, new_ys, dys)
        y_coeff = new_dys / binded_dx[1]

    y_coeff = y_coeff[:, np.newaxis].astype(np.float32)
    x_coeff = x_coeff[np.newaxis, :].astype(np.float32)

    x_coords = xs
    graph = np.stack([x_coords, h - (h / 4.) * x_coeff[0]], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (255, 255, 0), 2)

    y_coords = ys
    graph = np.stack([w - (w / 7.) * y_coeff[:, 0], y_coords], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (0, 255, 252), 2)
    cv2.line(image, (bind_point[0] + binded_dx[0], h - 1), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (255, 255, 0))
    cv2.line(image, (w-1, bind_point[1] + binded_dx[1]), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (0, 255, 252))
    cv2.circle(image, (bind_point[0], bind_point[1]), 8, (255, 255, 255), -1)
    cv2.circle(image, (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), 6, (0, 102, 255), -1)

    dx_ = dx * x_coeff * y_coeff

    dy_ = dy * x_coeff * y_coeff

    x_grid_ = x_grid - dx_
    y_grid_ = y_grid - dy_

    x_mask = r_mask
    x_image = cv2.remap(r_full, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x_mask = cv2.remap(x_mask, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    another_image = np.where(x_mask > 0.9, x_image, image)

    image = cv2.addWeighted(image, 0.5, another_image, 0.5, 0)

    return cv2.remap(image, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def binded_warp_cv_coloured_double(image, bind_point, binded_dx):
    h, w, c = image.shape
    ones = np.ones((h, w), dtype=np.float32)
    binded_dx = np.array(binded_dx).astype(np.int32)
    dx = ones * binded_dx[0]
    dy = ones * binded_dx[1]
    y_coeff = two_parabolas_across_dimension_np(h, bind_point[1])
    x_coeff = two_parabolas_across_dimension_np(w, bind_point[0])

    if binded_dx[0] != 0:
        dxs = x_coeff * binded_dx[0]
        new_xs = xs + dxs
        new_dxs = np.interp(xs, new_xs, dxs)
        x_coeff = new_dxs / binded_dx[0]

    if binded_dx[1] != 0:
        dys = y_coeff * binded_dx[1]
        new_ys = ys + dys
        new_dys = np.interp(ys, new_ys, dys)
        y_coeff = new_dys / binded_dx[1]

    y_coeff = y_coeff[:, np.newaxis].astype(np.float32)
    x_coeff = x_coeff[np.newaxis, :].astype(np.float32)

    x_coords = xs
    graph = np.stack([x_coords, h - (h / 4.) * x_coeff[0]], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (255, 255, 0), 2)

    y_coords = ys
    graph = np.stack([w - (w / 7.) * y_coeff[:, 0], y_coords], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (0, 255, 252), 2)
    cv2.line(image, (bind_point[0] + binded_dx[0], h - 1), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (255, 255, 0))
    cv2.line(image, (w-1, bind_point[1] + binded_dx[1]), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (0, 255, 252))
    cv2.circle(image, (bind_point[0], bind_point[1]), 8, (255, 255, 255), -1)
    cv2.circle(image, (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), 6, (0, 102, 255), -1)

    dx_ = dx * x_coeff * y_coeff

    dy_ = dy * x_coeff * y_coeff

    x_grid_ = x_grid - dx_
    y_grid_ = y_grid - dy_

    x_mask = r_mask
    x_image = cv2.remap(r_full, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    x_mask = cv2.remap(x_mask, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    another_image = np.where(x_mask > 0.9, x_image, image)

    another_image = cv2.remap(another_image, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    image = cv2.addWeighted(r_image, 0.5, image, 0.5, 0)

    image = cv2.addWeighted(image, 0.5, another_image, 0.5, 0)

    return cv2.remap(image, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def binded_warp_cv_coloured_image_grid(image, bind_point, binded_dx):
    h, w, c = image.shape
    ones = np.ones((h, w), dtype=np.float32)
    binded_dx = np.array(binded_dx).astype(np.int32)
    dx = ones * binded_dx[0]
    dy = ones * binded_dx[1]
    y_coeff = two_parabolas_across_dimension_np(h, bind_point[1])
    x_coeff = two_parabolas_across_dimension_np(w, bind_point[0])

    if binded_dx[0] != 0:
        dxs = x_coeff * binded_dx[0]
        new_xs = xs + dxs
        new_dxs = np.interp(xs, new_xs, dxs)
        x_coeff = new_dxs / binded_dx[0]

    if binded_dx[1] != 0:
        dys = y_coeff * binded_dx[1]
        new_ys = ys + dys
        new_dys = np.interp(ys, new_ys, dys)
        y_coeff = new_dys / binded_dx[1]

    y_coeff = y_coeff[:, np.newaxis].astype(np.float32)
    x_coeff = x_coeff[np.newaxis, :].astype(np.float32)

    x_coords = xs
    graph = np.stack([x_coords, h - (h / 4.) * x_coeff[0]], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (255, 255, 0), 2)

    y_coords = ys
    graph = np.stack([w - (w / 7.) * y_coeff[:, 0], y_coords], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (0, 255, 252), 2)
    cv2.line(image, (bind_point[0] + binded_dx[0], h - 1), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (255, 255, 0))
    cv2.line(image, (w-1, bind_point[1] + binded_dx[1]), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (0, 255, 252))
    cv2.circle(image, (bind_point[0], bind_point[1]), 8, (255, 255, 255), -1)
    cv2.circle(image, (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), 6, (0, 102, 255), -1)

    dx_ = dx * x_coeff * y_coeff
    dy_ = dy * x_coeff * y_coeff

    x_grid_ = x_grid - dx_
    y_grid_ = y_grid - dy_

    another_image = np.where(r_mask > 0.9, r_image, image)

    image = cv2.addWeighted(image, 0.5, another_image, 0.5, 0)

    return cv2.remap(image, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def binded_warp_cv_coloured_centered(image, bind_point, binded_dx):
    h, w, c = image.shape
    ones = np.ones((h, w), dtype=np.float32)
    binded_dx = np.array(binded_dx).astype(np.int32)
    dx = ones * binded_dx[0]
    dy = ones * binded_dx[1]
    y_coeff = two_parabolas_across_dimension_np(h, bind_point[1])
    x_coeff = two_parabolas_across_dimension_np(w, bind_point[0])

    if binded_dx[0] != 0:
        dxs = x_coeff * binded_dx[0]
        new_xs = xs - dxs
        new_dxs = np.interp(xs, new_xs, dxs)
        x_coeff = new_dxs / binded_dx[0]

    if binded_dx[1] != 0:
        dys = y_coeff * binded_dx[1]
        new_ys = ys - dys
        new_dys = np.interp(ys, new_ys, dys)
        y_coeff = new_dys / binded_dx[1]

    y_coeff = y_coeff[:, np.newaxis].astype(np.float32)
    x_coeff = x_coeff[np.newaxis, :].astype(np.float32)

    x_coords = np.arange(w)
    graph = np.stack([x_coords, h - (h / 4.) * x_coeff[0]], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (255, 255, 0), 2)

    y_coords = np.arange(h)
    graph = np.stack([w - (w / 7.) * y_coeff[:, 0], y_coords], axis=1).astype(np.int32)
    cv2.polylines(image, [graph], False, (0, 255, 252), 2)
    cv2.line(image, (bind_point[0] + binded_dx[0], h - 1), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (255, 255, 0))
    cv2.line(image, (w-1, bind_point[1] + binded_dx[1]), (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), (0, 255, 252))
    cv2.circle(image, (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), 8, (255, 255, 255), -1)
    cv2.circle(image, (bind_point[0] + binded_dx[0], bind_point[1] + binded_dx[1]), 6, (0, 102, 255), -1)

    another_image = np.where(r_mask > 0, r_image, image)
    image = cv2.addWeighted(image, 0.5, another_image, 0.5, 0)

    dx = dx * x_coeff * y_coeff
    dy = dy * x_coeff * y_coeff

    x_grid_ = x_grid + dx
    y_grid_ = y_grid + dy

    return cv2.remap(image, x_grid_, y_grid_, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


