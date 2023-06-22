""" Organize some frequently used visualization functions. """
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import seaborn as sns


# Plot junctions onto the image (return a separate copy)
def plot_junctions(input_image, junctions, junc_size=3, color=None):
    """
    input_image: can be 0~1 float or 0~255 uint8.
    junctions: Nx2 or 2xN np array.
    junc_size: the size of the plotted circles.
    """
    # Create image copy
    image = copy.copy(input_image)
    # Make sure the image is converted to 255 uint8
    if image.dtype == np.uint8:
        pass
    # A float type image ranging from 0~1
    elif image.dtype in [np.float32, np.float64, np.float]  and image.max() <= 2.:
        image = (image * 255.).astype(np.uint8)
    # A float type image ranging from 0.~255.
    elif image.dtype in [np.float32, np.float64, np.float] and image.mean() > 10.:
        image = image.astype(np.uint8)
    else:
        raise ValueError("[Error] Unknown image data type. Expect 0~1 float or 0~255 uint8.")

    # Check whether the image is single channel 
    if len(image.shape) == 2 or ((len(image.shape) == 3) and (image.shape[-1] == 1)):
        # Squeeze to H*W first
        image = image.squeeze()

        # Stack to channle 3
        image = np.concatenate([image[..., None] for _ in range(3)], axis=-1)

    # Junction dimensions should be N*2
    if not len(junctions.shape) == 2:
        raise ValueError("[Error] junctions should be 2-dim array.")

    # Always convert to N*2
    if junctions.shape[-1] != 2:
        if junctions.shape[0] == 2:
            junctions = junctions.T
        else:
            raise ValueError("[Error] At least one of the two dims should be 2.")
    
    # Round and convert junctions to int (and check the boundary)
    H, W = image.shape[:2]
    junctions = (np.round(junctions)).astype(np.int)
    junctions[junctions < 0] = 0 
    junctions[junctions[:, 0] >= H, 0] = H-1  # (first dim) max bounded by H-1
    junctions[junctions[:, 1] >= W, 1] = W-1  # (second dim) max bounded by W-1

    # Iterate through all the junctions
    num_junc = junctions.shape[0]
    if color is None:
        color = (0, 255., 0)
    for idx in range(num_junc):
        # Fetch one junction
        junc = junctions[idx, :]
        cv2.circle(image, tuple(np.flip(junc)), radius=junc_size, 
                    color=color, thickness=3)
    
    return image


# Plot line segements given junctions and line adjecent map
def plot_line_segments(input_image, junctions, line_map, junc_size=3, 
                       color=(0, 255., 0), line_width=1, plot_survived_junc=True):
    """
    input_image: can be 0~1 float or 0~255 uint8.
    junctions: Nx2 or 2xN np array.
    line_map: NxN np array
    junc_size: the size of the plotted circles.
    color: color of the line segments (can be string "random")
    line_width: width of the drawn segments.
    plot_survived_junc: whether we only plot the survived junctions.
    """
    # Create image copy
    image = copy.copy(input_image)
    # Make sure the image is converted to 255 uint8
    if image.dtype == np.uint8:
        pass
    # A float type image ranging from 0~1
    elif image.dtype in [np.float32, np.float64, np.float]  and image.max() <= 2.:
        image = (image * 255.).astype(np.uint8)
    # A float type image ranging from 0.~255.
    elif image.dtype in [np.float32, np.float64, np.float] and image.mean() > 10.:
        image = image.astype(np.uint8)
    else:
        raise ValueError("[Error] Unknown image data type. Expect 0~1 float or 0~255 uint8.")

    # Check whether the image is single channel 
    if len(image.shape) == 2 or ((len(image.shape) == 3) and (image.shape[-1] == 1)):
        # Squeeze to H*W first
        image = image.squeeze()

        # Stack to channle 3
        image = np.concatenate([image[..., None] for _ in range(3)], axis=-1)

    # Junction dimensions should be 2
    if not len(junctions.shape) == 2:
        raise ValueError("[Error] junctions should be 2-dim array.")

    # Always convert to N*2
    if junctions.shape[-1] != 2:
        if junctions.shape[0] == 2:
            junctions = junctions.T
        else:
            raise ValueError("[Error] At least one of the two dims should be 2.")
    
    # line_map dimension should be 2
    if not len(line_map.shape) == 2:
        raise ValueError("[Error] line_map should be 2-dim array.")

    # Color should be "random" or a list or tuple with length 3
    if color != "random":
        if not (isinstance(color, tuple) or isinstance(color, list)):
            raise ValueError("[Error] color should have type list or tuple.")
        else:
            if len(color) != 3:
                raise ValueError("[Error] color should be a list or tuple with length 3.")
    
    # Make a copy of the line_map
    line_map_tmp = copy.copy(line_map)

    # Parse line_map back to segment pairs
    segments = np.zeros([0, 4])
    for idx in range(junctions.shape[0]):
        # if no connectivity, just skip it
        if line_map_tmp[idx, :].sum() == 0:
            continue
        # record the line segment
        else:
            for idx2 in np.where(line_map_tmp[idx, :] == 1)[0]:
                p1 = np.flip(junctions[idx, :])     # Convert to xy format
                p2 = np.flip(junctions[idx2, :])    # Convert to xy format
                segments = np.concatenate((segments, np.array([p1[0], p1[1], p2[0], p2[1]])[None, ...]), axis=0)
                
                # Update line_map
                line_map_tmp[idx, idx2] = 0
                line_map_tmp[idx2, idx] = 0
    
    # Draw segment pairs
    for idx in range(segments.shape[0]):
        seg = np.round(segments[idx, :]).astype(np.int)
        # Decide the color
        if color != "random":
            color = tuple(color)
        else:
            color = tuple(np.random.rand(3,))
        cv2.line(image, tuple(seg[:2]), tuple(seg[2:]), color=color, thickness=line_width)

    # Also draw the junctions
    if not plot_survived_junc:
        num_junc = junctions.shape[0]
        for idx in range(num_junc):
            # Fetch one junction
            junc = junctions[idx, :]
            cv2.circle(image, tuple(np.flip(junc)), radius=junc_size, 
                    color=(0, 255., 0), thickness=3) 
    # Only plot the junctions which are part of a line segment
    else:
        for idx in range(segments.shape[0]):
            seg = np.round(segments[idx, :]).astype(np.int) # Already in HW format.
            cv2.circle(image, tuple(seg[:2]), radius=junc_size, 
                    color=(0, 255., 0), thickness=3)
            cv2.circle(image, tuple(seg[2:]), radius=junc_size, 
                    color=(0, 255., 0), thickness=3)
      
    return image


# Plot line segments given Nx4 or Nx2x2 line segments
def plot_line_segments_from_segments(input_image, line_segments, junc_size=3, 
                                     color=(0, 255., 0), line_width=1):
    # Create image copy
    image = copy.copy(input_image)
    # Make sure the image is converted to 255 uint8
    if image.dtype == np.uint8:
        pass
    # A float type image ranging from 0~1
    elif image.dtype in [np.float32, np.float64, np.float]  and image.max() <= 2.:
        image = (image * 255.).astype(np.uint8)
    # A float type image ranging from 0.~255.
    elif image.dtype in [np.float32, np.float64, np.float] and image.mean() > 10.:
        image = image.astype(np.uint8)
    else:
        raise ValueError("[Error] Unknown image data type. Expect 0~1 float or 0~255 uint8.")

    # Check whether the image is single channel 
    if len(image.shape) == 2 or ((len(image.shape) == 3) and (image.shape[-1] == 1)):
        # Squeeze to H*W first
        image = image.squeeze()

        # Stack to channle 3
        image = np.concatenate([image[..., None] for _ in range(3)], axis=-1)
    
    # Check the if line_segments are in (1) Nx4, or (2) Nx2x2.
    H, W, _ = image.shape
    # (1) Nx4 format
    if len(line_segments.shape) == 2 and line_segments.shape[-1] == 4:
        # Round to int32
        line_segments = line_segments.astype(np.int32)

        # Clip H dimension
        line_segments[:, 0] = np.clip(line_segments[:, 0], a_min=0, a_max=H-1)
        line_segments[:, 2] = np.clip(line_segments[:, 2], a_min=0, a_max=H-1)

        # Clip W dimension
        line_segments[:, 1] = np.clip(line_segments[:, 1], a_min=0, a_max=W-1)
        line_segments[:, 3] = np.clip(line_segments[:, 3], a_min=0, a_max=W-1)

        # Convert to Nx2x2 format
        line_segments = np.concatenate(
            [np.expand_dims(line_segments[:, :2], axis=1),       
            np.expand_dims(line_segments[:, 2:], axis=1)],
            axis=1
        )

    # (2) Nx2x2 format
    elif len(line_segments.shape) == 3 and line_segments.shape[-1] == 2:
        # Round to int32
        line_segments = line_segments.astype(np.int32)

        # Clip H dimension
        line_segments[:, :, 0] = np.clip(line_segments[:, :, 0], a_min=0, a_max=H-1)
        line_segments[:, :, 1] = np.clip(line_segments[:, :, 1], a_min=0, a_max=W-1)

    else:
        raise ValueError("[Error] line_segments should be either Nx4 or Nx2x2 in HW format.")

    # Draw segment pairs (all segments should be in HW format)
    image = image.copy()
    for idx in range(line_segments.shape[0]):
        seg = np.round(line_segments[idx, :, :]).astype(np.int32)
        # Decide the color
        if color != "random":
            color = tuple(color)
        else:
            color = tuple(np.random.rand(3,))
        cv2.line(image, tuple(np.flip(seg[0, :])), 
                        tuple(np.flip(seg[1, :])), 
                        color=color, thickness=line_width)

        # Also draw the junctions
        cv2.circle(image, tuple(np.flip(seg[0, :])), radius=junc_size, color=(0, 255., 0), thickness=3)
        cv2.circle(image, tuple(np.flip(seg[1, :])), radius=junc_size, color=(0, 255., 0), thickness=3)
    
    return image


# Additional functions to visualize multiple images at the same time,
# e.g. for line matching
def plot_images(imgs, titles=None, cmaps='gray', dpi=100, size=6, pad=.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n
    figsize = (size*n, size*3/4) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors='lime', ps=4):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps, zorder=2)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps, zorder=2)


def plot_lines(lines, line_colors='orange', point_colors='cyan',
               ps=4, lw=2, indices=(0, 1)):
    """Plot lines and endpoints for existing images.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float pixels.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(lines)
    if not isinstance(point_colors, list):
        point_colors = [point_colors] * len(lines)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, l, lc, pc in zip(axes, lines, line_colors, point_colors):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D((l[i, 0, 0], l[i, 1, 0]),
                                           (l[i, 0, 1], l[i, 1, 1]),
                                           zorder=1, c=lc, linewidth=lw)
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 0], pts[:, 1],
                  c=pc, s=ps, linewidths=0, zorder=2)


def plot_line_matches(kpts0, kpts1, color=None, lw=1.5, indices=(0, 1), a=1.):
    """Plot matches for a pair of existing images, parametrized by their middle point.
    Args:
        kpts0, kpts1: corresponding middle points of the lines of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [matplotlib.lines.Line2D(
            (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
            zorder=1, transform=fig.transFigure, c=color[i], linewidth=lw,
            alpha=a)
            for i in range(len(kpts0))]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)


def plot_color_line_matches(lines, correct_matches=None,
                            lw=2, indices=(0, 1)):
    """Plot line matches for existing images with multiple colors.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        correct_matches: bool array of size (N,) indicating correct matches.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    n_lines = len(lines[0])
    colors = sns.color_palette('husl', n_colors=n_lines)
    np.random.shuffle(colors)
    alphas = np.ones(n_lines)
    # If correct_matches is not None, display wrong matches with a low alpha
    if correct_matches is not None:
        alphas[~np.array(correct_matches)] = 0.2

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, l in zip(axes, lines):
        # Transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        endpoint0 = transFigure.transform(a.transData.transform(l[:, 0]))
        endpoint1 = transFigure.transform(a.transData.transform(l[:, 1]))
        fig.lines += [matplotlib.lines.Line2D(
            (endpoint0[i, 0], endpoint1[i, 0]),
            (endpoint0[i, 1], endpoint1[i, 1]),
            zorder=1, transform=fig.transFigure, c=colors[i],
            alpha=alphas[i], linewidth=lw) for i in range(n_lines)]


def plot_color_lines(lines, correct_matches, wrong_matches,
                     lw=2, indices=(0, 1)):
    """Plot line matches for existing images with multiple colors:
    green for correct matches, red for wrong ones, and blue for the rest.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        correct_matches: list of bool arrays of size N with correct matches.
        wrong_matches: list of bool arrays of size (N,) with correct matches.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    # palette = sns.color_palette()
    palette = sns.color_palette("hls", 8)
    blue = palette[5]  # palette[0]
    red = palette[0]  # palette[3]
    green = palette[2]  # palette[2]
    colors = [np.array([blue] * len(l)) for l in lines]
    for i, c in enumerate(colors):
        c[np.array(correct_matches[i])] = green
        c[np.array(wrong_matches[i])] = red

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, l, c in zip(axes, lines, colors):
        # Transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        endpoint0 = transFigure.transform(a.transData.transform(l[:, 0]))
        endpoint1 = transFigure.transform(a.transData.transform(l[:, 1]))
        fig.lines += [matplotlib.lines.Line2D(
            (endpoint0[i, 0], endpoint1[i, 0]),
            (endpoint0[i, 1], endpoint1[i, 1]),
            zorder=1, transform=fig.transFigure, c=c[i],
            linewidth=lw) for i in range(len(l))]


def plot_subsegment_matches(lines, subsegments, lw=2, indices=(0, 1)):
    """ Plot line matches for existing images with multiple colors and
        highlight the actually matched subsegments.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        subsegments: list of ndarrays of size (N, 2, 2).
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    n_lines = len(lines[0])
    colors = sns.cubehelix_palette(start=2, rot=-0.2, dark=0.3, light=.7,
                                   gamma=1.3, hue=1, n_colors=n_lines)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, l, ss in zip(axes, lines, subsegments):
        # Transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()

        # Draw full line
        endpoint0 = transFigure.transform(a.transData.transform(l[:, 0]))
        endpoint1 = transFigure.transform(a.transData.transform(l[:, 1]))
        fig.lines += [matplotlib.lines.Line2D(
            (endpoint0[i, 0], endpoint1[i, 0]),
            (endpoint0[i, 1], endpoint1[i, 1]),
            zorder=1, transform=fig.transFigure, c='red',
            alpha=0.7, linewidth=lw) for i in range(n_lines)]

        # Draw matched subsegment
        endpoint0 = transFigure.transform(a.transData.transform(ss[:, 0]))
        endpoint1 = transFigure.transform(a.transData.transform(ss[:, 1]))
        fig.lines += [matplotlib.lines.Line2D(
            (endpoint0[i, 0], endpoint1[i, 0]),
            (endpoint0[i, 1], endpoint1[i, 1]),
            zorder=1, transform=fig.transFigure, c=colors[i],
            alpha=1, linewidth=lw) for i in range(n_lines)]