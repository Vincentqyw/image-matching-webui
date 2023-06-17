import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import collections as mplcollections
from matplotlib import colors as mcolors
from torch_dimcheck import dimchecked

from disk import MatchedPairs

class MultiFigure:
    @dimchecked
    def __init__(
        self,
        image1: ['H', 'W', 'C'],
        image2: ['H', 'W', 'C'],
        grid=None,
        vertical=False,
    ):
        assert image1.shape == image2.shape
        h, w, c = image1.shape

        cat_dim = 0 if vertical else 1
        images = torch.cat([image1, image2], dim=cat_dim)

        figsize = (20, 40) if vertical else (40, 20)

        self.fig, self._ax = plt.subplots(
            figsize=figsize,
            frameon=False,
            constrained_layout=True
        )
        self._ax.imshow(images)
        xmax = w
        ymax = h
        if vertical:
            ymax *= 2
        else:
            xmax *= 2

        self._ax.set_xlim(0, xmax)
        self._ax.set_ylim(ymax, 0)

        if grid is None:
            self._ax.axis('off')
        else:
            self._ax.set_xticks(np.arange(0, xmax, grid))
            self._ax.set_yticks(np.arange(0, ymax, grid))
            self._ax.grid()

        if vertical:
            self.offset = torch.tensor([0, h])
        else:
            self.offset = torch.tensor([w, 0])

    @dimchecked
    def mark_xy(
        self,
        xy1: [2, 'N'],
        xy2: [2, 'N'],
        color='green',
        lines=True,
        marks=True,
        plot_n=None,
        linewidth=None,
        marker_size=None,
    ):
        xy2 = xy2 + self.offset.reshape(2, 1)

        xys = torch.stack([xy1.T, xy2.T], dim=1)

        if plot_n is not None:
            if xys.shape[0] > plot_n:
                ixs = torch.linspace(0, xys.shape[0]-1, plot_n).to(torch.int64)
                xys = xys[ixs, :]

        if lines:
            if color is not None:
                # LineCollection requires an rgb tuple
                color = mcolors.to_rgb(color)

            # yx convention
            plot = mplcollections.LineCollection(
                xys.numpy(),
                color=color,
                linewidth=linewidth
            )
            self._ax.add_collection(plot)
        else:
            plot = None

        if marks:
            self._ax.scatter(
                xys[:, :, 0].numpy().flatten(),
                xys[:, :, 1].numpy().flatten(),
                marker='o',
                c='white',
                edgecolor='black',
                s=marker_size,
            )
    
        return plot
