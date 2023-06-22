"""
2D visualization primitives based on Matplotlib.

1) Plot images with `plot_images`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Optionally: save a .png or .pdf plot (nice in papers!) with `save_plot`.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import flow_vis


def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None]*2
    c = x*np.array([[0, 1., 0]]) + (2-x)*np.array([[1., 0, 0]])
    return np.clip(c, 0, 1)


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
        
        
def plot_lines(lines, line_colors='orange', point_color='cyan',
               ps=4, lw=2, indices=(0, 1), alpha=1):
    """ Plot lines and endpoints for existing images.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        line_colors: string, or list of list of tuples (one for per line).
        point_color: unique color for all endpoints.
        ps: size of the keypoints as float pixels.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
        alpha: alpha transparency.
    """
    if not isinstance(line_colors, list):
        line_colors = [[line_colors] * len(l) for l in lines]
    for i in range(len(lines)):
        if ((not isinstance(line_colors[i], list))
            and (not isinstance(line_colors[i], np.ndarray))):
            line_colors[i] = [line_colors[i]] * len(lines[i])

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, l, lc in zip(axes, lines, line_colors):
        for i in range(len(l)):
            line = matplotlib.lines.Line2D(
                (l[i, 0, 0], l[i, 1, 0]), (l[i, 0, 1], l[i, 1, 1]),
                zorder=1, c=lc[i], linewidth=lw, alpha=alpha)
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 0], pts[:, 1], c=point_color, s=ps,
                  linewidths=0, zorder=2, alpha=alpha)

        
def plot_vp(lines, vp_labels, lw=2, indices=(0, 1)):
    """ Plot the vanishing directions of the lines, given the vp labels.
    Lines labelled with -1 are ignored.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        vp_labels: list of labels indicating the corresponding vp.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    num_labels = np.amax([np.amax(vp) for vp in vp_labels if len(vp) > 0]) + 1
    colors = sns.color_palette("hls", num_labels)
    
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, l, vp in zip(axes, lines, vp_labels):
        for i in range(len(l)):
            if vp[i] == -1:
                continue
            line = matplotlib.lines.Line2D(
                (l[i, 0, 0], l[i, 1, 0]), (l[i, 0, 1], l[i, 1, 1]),
                zorder=1, c=colors[vp[i]], linewidth=lw)
            a.add_line(line)


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


def get_flow_vis(df, ang, line_neighborhood=5):
    norm = line_neighborhood + 1 - np.clip(df, 0, line_neighborhood)
    flow_uv = np.stack([norm * np.cos(ang), norm * np.sin(ang)], axis=-1)
    flow_img = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    return flow_img


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
