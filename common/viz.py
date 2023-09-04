import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=5, pad=0.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n
    # figsize = (size*n, size*3/4) if size is not None else None
    figsize = (size * n, size * 6 / 5) if size is not None else None
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
    return fig


def plot_color_line_matches(lines, correct_matches=None, lw=2, indices=(0, 1)):
    """Plot line matches for existing images with multiple colors.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        correct_matches: bool array of size (N,) indicating correct matches.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    n_lines = len(lines[0])
    colors = sns.color_palette("husl", n_colors=n_lines)
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
        fig.lines += [
            matplotlib.lines.Line2D(
                (endpoint0[i, 0], endpoint1[i, 0]),
                (endpoint0[i, 1], endpoint1[i, 1]),
                zorder=1,
                transform=fig.transFigure,
                c=colors[i],
                alpha=alphas[i],
                linewidth=lw,
            )
            for i in range(n_lines)
        ]

    return fig


def make_matching_figure(
    img0,
    img1,
    mkpts0,
    mkpts1,
    color,
    titles=None,
    kpts0=None,
    kpts1=None,
    text=[],
    dpi=75,
    path=None,
    pad=0,
):
    # draw image pair
    # assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)  # , cmap='gray')
    axes[1].imshow(img1)  # , cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
        if titles is not None:
            axes[i].set_title(titles[i])

    plt.tight_layout(pad=pad)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=5)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=5)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c=color[i],
                linewidth=2,
            )
            for i in range(len(mkpts0))
        ]

        # freeze the axes to prevent the transform to change
        axes[0].autoscale(enable=False)
        axes[1].autoscale(enable=False)

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color[..., :3], s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color[..., :3], s=4)

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack(
            [2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1
        ),
        0,
        1,
    )


np.random.seed(1995)
color_map = np.arange(100)
np.random.shuffle(color_map)


def fig2im(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype="u1")
    im = buf_ndarray.reshape(h, w, 3)
    return im


def draw_matches(
    mkpts0, mkpts1, img0, img1, conf, titles=None, dpi=150, path=None, pad=0.5
):
    thr = 5e-4
    thr = 0.5
    color = error_colormap(conf, thr, alpha=0.1)
    text = [
        f"image name",
        f"#Matches: {len(mkpts0)}",
    ]
    if path:
        fig2im(
            make_matching_figure(
                img0,
                img1,
                mkpts0,
                mkpts1,
                color,
                titles=titles,
                text=text,
                path=path,
                dpi=dpi,
                pad=pad,
            )
        )
    else:
        return fig2im(
            make_matching_figure(
                img0,
                img1,
                mkpts0,
                mkpts1,
                color,
                titles=titles,
                text=text,
                pad=pad,
                dpi=dpi,
            )
        )


def draw_image_pairs(img0, img1, text=[], dpi=75, path=None, pad=0.5):
    # draw image pair
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)  # , cmap='gray')
    axes[1].imshow(img1)  # , cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=pad)

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig2im(fig)
