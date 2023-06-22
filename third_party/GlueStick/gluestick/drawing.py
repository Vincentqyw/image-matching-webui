import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_images(imgs, titles=None, cmaps='gray', dpi=100, pad=.5,
                adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={'width_ratios': ratios})
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
    return ax


def plot_keypoints(kpts, colors='lime', ps=4, alpha=1):
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
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, alpha=alpha, linewidths=0)


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
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_lines(lines, line_colors='orange', point_colors='cyan',
               ps=4, lw=2, alpha=1., indices=(0, 1)):
    """ Plot lines and endpoints for existing images.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float pixels.
        lw: line width as float pixels.
        alpha: transparency of the points and lines.
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
                                           zorder=1, c=lc, linewidth=lw,
                                           alpha=alpha)
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 0], pts[:, 1],
                  c=pc, s=ps, linewidths=0, zorder=2, alpha=alpha)


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
