import typing
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns

from ..hloc.utils.viz import add_text, plot_keypoints

np.random.seed(1995)
color_map = np.arange(100)
np.random.shuffle(color_map)

# Fixed top margin for titles (inches) so text never gets clipped
_TITLE_MARGIN_INCHES = 0.25
# Small gap between subplots
_WSPACE = 0.005
# Figure width clamp (inches): at 150 dpi, 14-16 inches is 2100-2400 px, enough
# for a retina browser but bounded, so a 12 MP upload does not turn into a
# 6000 px figure.
_MIN_FIG_WIDTH = 14.0
_MAX_FIG_WIDTH = 16.0
# Longest panel side at those caps (portrait pair); pre-resizing uploads to this
# keeps Matplotlib from resampling multi-megapixel sources.
_DISPLAY_MAX_SIDE = 1600


def _figure_size_for_images(
    images: List[np.ndarray],
    dpi: int,
    ncols: int = 1,
    title_margin_inches: float = _TITLE_MARGIN_INCHES,
) -> Tuple[Tuple[float, float], float, List[float]]:
    """Compute figure size, axes ``top`` fraction, and width ratios.

    Each subplot's width is proportional to its image's aspect ratio so
    every image fills its subplot without side-whitespace, even when the
    two images have different dimensions.
    """
    ratios = [img.shape[1] / img.shape[0] for img in images]  # image ARs
    # Natural size — images at 1:1 pixel mapping.
    natural_w = max(img.shape[0] for img in images) / dpi * sum(ratios)
    fig_w = min(max(natural_w, _MIN_FIG_WIDTH), _MAX_FIG_WIDTH)
    axes_h = fig_w / sum(ratios)  # keep subplot AR == image AR

    fig_h = axes_h + title_margin_inches
    return (fig_w, fig_h), axes_h / fig_h, ratios


def fit_for_display(img: np.ndarray, *arrays):
    """Downscale ``img`` (and coordinate arrays) to the size it is drawn at.

    Matplotlib would otherwise resample a full-resolution upload into the
    panel, which costs more than everything else in the render.  Images no
    larger than ``_DISPLAY_MAX_SIDE`` are returned untouched.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= _DISPLAY_MAX_SIDE:
        return (img, *arrays)
    scale = _DISPLAY_MAX_SIDE / side
    img = cv2.resize(
        img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
    )
    return (img, *(None if a is None else a * scale for a in arrays))


def _stroked_text(ax, x, y, text, fontsize=12, color="w", lw=2, **kwargs):
    """Add text with a dark outline so it is readable on any background."""
    t = ax.text(
        x, y, text, fontsize=fontsize, color=color, transform=ax.transAxes, **kwargs
    )
    t.set_path_effects(
        [
            path_effects.Stroke(linewidth=lw, foreground="k"),
            path_effects.Normal(),
        ]
    )
    return t


def _add_figure_lines(fig, pts0, pts1, colors, linewidths):
    """Draw segments ``pts0``-``pts1`` (display coords) across the figure."""
    to_fig = fig.transFigure.inverted()
    fig.add_artist(
        LineCollection(
            np.stack([to_fig.transform(pts0), to_fig.transform(pts1)], axis=1),
            colors=colors,
            linewidths=linewidths,
            transform=fig.transFigure,
            zorder=1,
        )
    )


def plot_images(
    imgs: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cmaps: Union[str, List[str]] = "gray",
    dpi: int = 150,
    size: Optional[int] = None,
    pad: float = 0.0,
) -> plt.Figure:
    """Plot a set of images horizontally with minimal whitespace (paper-style).

    Figure size is computed from image pixel dimensions.  A fixed top margin
    (0.25") is reserved for titles so they never obscure image content.
    """
    n = len(imgs)
    if not isinstance(cmaps, list):
        cmaps = [cmaps] * n
    if isinstance(titles, str):
        titles = [titles]

    figsize, top, width_ratios = _figure_size_for_images(imgs, dpi, ncols=n)
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": width_ratios}
    )

    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        ax[i].margins(0, 0)
        for spine in ax[i].spines.values():
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i], pad=2, fontsize=12)

    plt.subplots_adjust(
        left=0.005, right=0.995, bottom=0, top=top, wspace=_WSPACE, hspace=0
    )
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
    rgba = np.column_stack([np.asarray(colors), alphas])
    for a, l in zip(axes, lines):  # noqa: E741
        _add_figure_lines(
            fig,
            a.transData.transform(l[:, 0]),
            a.transData.transform(l[:, 1]),
            rgba,
            lw,
        )

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
    """Draw image pair with matches (paper-style, minimal whitespace).

    Figure size is computed from image pixel dimensions.  A fixed top margin
    is reserved for titles.  Overlay text has a dark outline for readability.
    """
    img0, mkpts0, kpts0 = fit_for_display(img0, mkpts0, kpts0)
    img1, mkpts1, kpts1 = fit_for_display(img1, mkpts1, kpts1)
    figsize, top, width_ratios = _figure_size_for_images([img0, img1], dpi, ncols=2)

    # draw image pair
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": width_ratios}
    )
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        axes[i].set_axis_off()
        axes[i].margins(0, 0)
        for spine in axes[i].spines.values():
            spine.set_visible(False)
        if titles is not None:
            axes[i].set_title(titles[i], pad=2, fontsize=12)

    plt.subplots_adjust(
        left=0.005, right=0.995, bottom=0, top=top, wspace=_WSPACE, hspace=0
    )

    # Draw all keypoints (small white dots)
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=4, edgecolors="none")
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=4, edgecolors="none")

    # Draw match lines + matched-keypoint scatter
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0 and mkpts0.shape == mkpts1.shape:
        fig.canvas.draw()
        _add_figure_lines(
            fig,
            axes[0].transData.transform(mkpts0),
            axes[1].transData.transform(mkpts1),
            color,
            1.5,
        )

        axes[0].autoscale(enable=False)
        axes[1].autoscale(enable=False)

        # Matched keypoints — small colored dots
        axes[0].scatter(
            mkpts0[:, 0], mkpts0[:, 1], c=color[..., :3], s=5, edgecolors="none"
        )
        axes[1].scatter(
            mkpts1[:, 0], mkpts1[:, 1], c=color[..., :3], s=5, edgecolors="none"
        )

    # Overlay match count with stroke so it's readable on any background
    if text:
        _stroked_text(
            axes[0], 0.01, 0.99, "\n".join(text), fontsize=12, va="top", ha="left"
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
    """Convert a matplotlib figure to a numpy RGB array.

    Uses an in-memory PNG buffer so the output is immune to HiDPI
    (Retina) canvas-vs-buffer mismatches.  ``bbox_inches=None`` keeps the
    exact figure size so that figures with different title lengths stay
    pixel-identical in width.  Closes ``fig``.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches=None,
        pad_inches=0,
        dpi=fig.dpi,
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    rgba = plt.imread(buf)
    plt.close(fig)
    if rgba.shape[-1] == 4:
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    return (rgba * 255).astype(np.uint8)


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
    pad: float = 0.0,
) -> np.ndarray:
    """Draw matches between two images."""
    thr = 0.5
    color = error_colormap(1 - conf, thr, alpha=0.1)
    if texts:
        text = list(texts)
    else:
        text = [f"#Matches: {len(mkpts0)}"]
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
                dpi=dpi,
            )
        )


def draw_image_pairs(
    img0: np.ndarray,
    img1: np.ndarray,
    text: List[str] = [],
    dpi: int = 75,
    path: Optional[str] = None,
    pad: float = 0.0,
) -> np.ndarray:
    """Draw image pair horizontally with minimal whitespace (paper-style).

    Figure size is computed from image pixel dimensions.  A fixed top margin
    is reserved for overlay text.
    """
    figsize, top, width_ratios = _figure_size_for_images([img0, img1], dpi, ncols=2)

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": width_ratios}
    )
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        axes[i].set_axis_off()
        axes[i].margins(0, 0)
        for spine in axes[i].spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(
        left=0.005, right=0.995, bottom=0, top=top, wspace=_WSPACE, hspace=0
    )

    # Stroked overlay text so it's readable on any background
    if text:
        _stroked_text(
            axes[0], 0.01, 0.99, "\n".join(text), fontsize=12, va="top", ha="left"
        )

    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig2im(fig)


def display_keypoints(pred: dict, titles: List[str] = []):
    img0, kpts0 = fit_for_display(pred["image0_orig"], pred.get("keypoints0_orig"))
    img1, kpts1 = fit_for_display(pred["image1_orig"], pred.get("keypoints1_orig"))
    fig = plot_images([img0, img1], titles=titles)
    if kpts0 is not None and kpts1 is not None:
        plot_keypoints([kpts0, kpts1], ps=4)
        add_text(0, f"# keypoints0: {len(kpts0)} \n# keypoints1: {len(kpts1)}", fs=12)
    return fig2im(fig)


def display_matches(
    pred: Dict[str, np.ndarray],
    titles: List[str] = [],
    texts: List[str] = [],
    dpi: int = 150,
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
    elif (
        "line0_orig" in pred
        and "line1_orig" in pred
        and pred["line0_orig"] is not None
        and pred["line1_orig"] is not None
        # and (tag == "LINES_RAW" or tag == "LINES_RANSAC")
    ):
        # lines — fit images and their coordinates at the same scale
        img0, mtlines0, mkpts0 = fit_for_display(
            img0, pred["line0_orig"], pred.get("line_keypoints0_orig")
        )
        img1, mtlines1, mkpts1 = fit_for_display(
            img1, pred["line1_orig"], pred.get("line_keypoints1_orig")
        )
        num_inliers = len(mtlines0)
        plot_images(
            [img0.squeeze(), img1.squeeze()],
            ["Image 0 - matched lines", "Image 1 - matched lines"],
        )
        # plot_color_line_matches annotates the current figure
        fig = fig2im(plot_color_line_matches([mtlines0, mtlines1], lw=2))

        # keypoints
        if mkpts0 is not None and mkpts1 is not None:
            num_inliers = len(mkpts0)
            mconf = pred["mconf"] if "mconf" in pred else np.ones(len(mkpts0))
            fig_mkpts = draw_matches_core(mkpts0, mkpts1, img0, img1, mconf, dpi=dpi)
            h, w = fig_mkpts.shape[:2]
            fig = np.concatenate([fig_mkpts, cv2.resize(fig, (w, h))], axis=0)
    return fig, num_inliers
