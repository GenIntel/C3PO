def pxl2d_2_pxl2d_normalized(grid_xy, H=None, W=None):
    # ensure normalize pxlcoords is no inplace
    grid_xy = grid_xy.clone()

    if grid_xy.dims() < 4:
        no_batch = True
        grid_xy = grid_xy[None,]
    else:
        no_batch = False

    if H is None or W is None:
        B, C, H, W = grid_xy.shape

    grid_xy[:, 0] = grid_xy[:, 0] / (W - 1.0) * 2.0 - 1.0
    grid_xy[:, 1] = grid_xy[:, 1] / (H - 1.0) * 2.0 - 1.0

    if no_batch:
        grid_xy = grid_xy[0]

    return grid_xy

