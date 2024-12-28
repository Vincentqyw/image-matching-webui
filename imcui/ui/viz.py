import sys
import typing
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(str(Path(__file__).parents[1]))

from hloc.utils.viz import add_text, plot_keypoints

np.random.seed(1995)
color_map = np.arange(100)
np.random.shuffle(color_map)


def plot_images(
    imgs: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cmaps: Union[str, List[str]] = "gray",
    dpi: int = 100,
    size: Optional[int] = 5,
    pad: float = 0.5,
) -> plt.Figure:
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images. If a single string is given,
            it is used for all images.
        dpi: DPI of the figure.
        size: figure size in inches (width). If not provided, the figure
            size is determined automatically.
        pad: padding between subplots, in inches.
    Returns:
        The created figure.
    """
    n = len(imgs)
    if not isinstance(cmaps, list):
        cmaps = [cmaps] * n
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


def plot_color_line_matches(
    lines: List[np.ndarray],
    correct_matches: Optional[np.ndarray] = None,
    lw: float = 2.0,
    indices: Tuple[int, int] = (0, 1),
) -> matplotlib.figure.Figure:
    """Plot line matches for existing images with multiple colors.

    Args:
        lines: List of ndarrays of size (N, 2, 2) representing line segments.
        correct_matches: Optional bool array of size (N,) indicating correct
            matches. If not None, display wrong matches with a low alpha.
        lw: Line width as float pixels.
        indices: Indices of the images to draw the matches on.

    Returns:
        The modified matplotlib figure.
    """
    n_lines = lines[0].shape[0]
    colors = sns.color_palette("husl", n_colors=n_lines)
    np.random.shuffle(colors)
    alphas = np.ones(n_lines)
    if correct_matches is not None:
        alphas[~np.array(correct_matches)] = 0.2

    fig = plt.gcf()
    ax = typing.cast(List[matplotlib.axes.Axes], fig.axes)
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines
    for a, l in zip(axes, lines):  # noqa: E741
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
    img0: np.ndarray,
    img1: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    color: np.ndarray,
    titles: Optional[List[str]] = None,
    kpts0: Optional[np.ndarray] = None,
    kpts1: Optional[np.ndarray] = None,
    text: List[str] = [],
    dpi: int = 75,
    path: Optional[Path] = None,
    pad: float = 0.0,
) -> Optional[plt.Figure]:
    """Draw image pair with matches.

    Args:
        img0: image0 as HxWx3 numpy array.
        img1: image1 as HxWx3 numpy array.
        mkpts0: matched points in image0 as Nx2 numpy array.
        mkpts1: matched points in image1 as Nx2 numpy array.
        color: colors for the matches as Nx4 numpy array.
        titles: titles for the two subplots.
        kpts0: keypoints in image0 as Kx2 numpy array.
        kpts1: keypoints in image1 as Kx2 numpy array.
        text: list of strings to display in the top-left corner of the image.
        dpi: dots per inch of the saved figure.
        path: if not None, save the figure to this path.
        pad: padding around the image as a fraction of the image size.

    Returns:
        The matplotlib Figure object if path is None.
    """
    # draw image pair
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
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0 and mkpts0.shape == mkpts1.shape:
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


def error_colormap(err: np.ndarray, thr: float, alpha: float = 1.0) -> np.ndarray:
    """
    Create a colormap based on the error values.

    Args:
        err: Error values as a numpy array of shape (N,).
        thr: Threshold value for the error.
        alpha: Alpha value for the colormap, between 0 and 1.

    Returns:
        Colormap as a numpy array of shape (N, 4) with values in [0, 1].
    """
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1),
        0,
        1,
    )


def fig2im(fig: matplotlib.figure.Figure) -> np.ndarray:
    """
    Convert a matplotlib figure to a numpy array with RGB values.

    Args:
        fig: A matplotlib figure.

    Returns:
        A numpy array with shape (height, width, 3) and dtype uint8 containing
        the RGB values of the figure.
    """
    fig.canvas.draw()
    (width, height) = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype="u1")
    return buf_ndarray.reshape(height, width, 3)


def draw_matches_core(
    mkpts0: List[np.ndarray],
    mkpts1: List[np.ndarray],
    img0: np.ndarray,
    img1: np.ndarray,
    conf: np.ndarray,
    titles: Optional[List[str]] = None,
    texts: Optional[List[str]] = None,
    dpi: int = 150,
    path: Optional[str] = None,
    pad: float = 0.5,
) -> np.ndarray:
    """
    Draw matches between two images.

    Args:
        mkpts0: List of matches from the first image, with shape (N, 2)
        mkpts1: List of matches from the second image, with shape (N, 2)
        img0: First image, with shape (H, W, 3)
        img1: Second image, with shape (H, W, 3)
        conf: Confidence values for the matches, with shape (N,)
        titles: Optional list of title strings for the plot
        dpi: DPI for the saved image
        path: Optional path to save the image to. If None, the image is not saved.
        pad: Padding between subplots

    Returns:
        The figure as a numpy array with shape (height, width, 3) and dtype uint8
        containing the RGB values of the figure.
    """
    thr = 0.5
    color = error_colormap(1 - conf, thr, alpha=0.1)
    text = [
        # "image name",
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


def draw_image_pairs(
    img0: np.ndarray,
    img1: np.ndarray,
    text: List[str] = [],
    dpi: int = 75,
    path: Optional[str] = None,
    pad: float = 0.5,
) -> np.ndarray:
    """Draw image pair horizontally.

    Args:
        img0: First image, with shape (H, W, 3)
        img1: Second image, with shape (H, W, 3)
        text: List of strings to print. Each string is a new line.
        dpi: DPI of the figure.
        path: Path to save the image to. If None, the image is not saved and
            the function returns the figure as a numpy array with shape
            (height, width, 3) and dtype uint8 containing the RGB values of the
            figure.
        pad: Padding between subplots

    Returns:
        The figure as a numpy array with shape (height, width, 3) and dtype uint8
        containing the RGB values of the figure, or None if path is not None.
    """
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


def display_keypoints(pred: dict, titles: List[str] = []):
    img0 = pred["image0_orig"]
    img1 = pred["image1_orig"]
    output_keypoints = plot_images([img0, img1], titles=titles, dpi=300)
    if "keypoints0_orig" in pred.keys() and "keypoints1_orig" in pred.keys():
        plot_keypoints([pred["keypoints0_orig"], pred["keypoints1_orig"]])
        text = (
            f"# keypoints0: {len(pred['keypoints0_orig'])} \n"
            + f"# keypoints1: {len(pred['keypoints1_orig'])}"
        )
        add_text(0, text, fs=15)
    output_keypoints = fig2im(output_keypoints)
    return output_keypoints


def display_matches(
    pred: Dict[str, np.ndarray],
    titles: List[str] = [],
    texts: List[str] = [],
    dpi: int = 300,
    tag: str = "KPTS_RAW",  # KPTS_RAW, KPTS_RANSAC, LINES_RAW, LINES_RANSAC,
) -> Tuple[np.ndarray, int]:
    """
    Displays the matches between two images.

    Args:
        pred: Dictionary containing the original images and the matches.
        titles: Optional titles for the plot.
        dpi: Resolution of the plot.

    Returns:
        The resulting concatenated plot and the number of inliers.
    """
    img0 = pred["image0_orig"]
    img1 = pred["image1_orig"]
    num_inliers = 0
    KPTS0_KEY = None
    KPTS1_KEY = None
    confid = None
    if tag == "KPTS_RAW":
        KPTS0_KEY = "mkeypoints0_orig"
        KPTS1_KEY = "mkeypoints1_orig"
        if "mconf" in pred:
            confid = pred["mconf"]
    elif tag == "KPTS_RANSAC":
        KPTS0_KEY = "mmkeypoints0_orig"
        KPTS1_KEY = "mmkeypoints1_orig"
        if "mmconf" in pred:
            confid = pred["mmconf"]
    else:
        # TODO: LINES_RAW, LINES_RANSAC
        raise ValueError(f"Unknown tag: {tag}")
    # draw raw matches
    if (
        KPTS0_KEY in pred
        and KPTS1_KEY in pred
        and pred[KPTS0_KEY] is not None
        and pred[KPTS1_KEY] is not None
    ):  # draw ransac matches
        mkpts0 = pred[KPTS0_KEY]
        mkpts1 = pred[KPTS1_KEY]
        num_inliers = len(mkpts0)
        if confid is None:
            confid = np.ones(len(mkpts0))
        fig_mkpts = draw_matches_core(
            mkpts0,
            mkpts1,
            img0,
            img1,
            confid,
            dpi=dpi,
            titles=titles,
            texts=texts,
        )
        fig = fig_mkpts
    # TODO: draw lines
    if (
        "line0_orig" in pred
        and "line1_orig" in pred
        and pred["line0_orig"] is not None
        and pred["line1_orig"] is not None
        and (tag == "LINES_RAW" or tag == "LINES_RANSAC")
    ):
        # lines
        mtlines0 = pred["line0_orig"]
        mtlines1 = pred["line1_orig"]
        num_inliers = len(mtlines0)
        fig_lines = plot_images(
            [img0.squeeze(), img1.squeeze()],
            ["Image 0 - matched lines", "Image 1 - matched lines"],
            dpi=300,
        )
        fig_lines = plot_color_line_matches([mtlines0, mtlines1], lw=2)
        fig_lines = fig2im(fig_lines)

        # keypoints
        mkpts0 = pred.get("line_keypoints0_orig")
        mkpts1 = pred.get("line_keypoints1_orig")
        fig = None
        if mkpts0 is not None and mkpts1 is not None:
            num_inliers = len(mkpts0)
            if "mconf" in pred:
                mconf = pred["mconf"]
            else:
                mconf = np.ones(len(mkpts0))
            fig_mkpts = draw_matches_core(mkpts0, mkpts1, img0, img1, mconf, dpi=300)
            fig_lines = cv2.resize(fig_lines, (fig_mkpts.shape[1], fig_mkpts.shape[0]))
            fig = np.concatenate([fig_mkpts, fig_lines], axis=0)
        else:
            fig = fig_lines
    return fig, num_inliers
