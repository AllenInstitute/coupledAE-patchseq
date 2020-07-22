import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set(style='white')


def contingency(a, b, unique_a, unique_b):
    """Populate contingency matrix. Rows and columns are not normalized in any way.
    
    Args:
        a (np.array): labels
        b (np.array): labels
        unique_a (np.array): unique list of labels. Can have more entries than np.unique(a)
        unique_b (np.array): unique list of labels. Can have more entries than np.unique(b)

    Returns:
        C (np.array): contingency matrix.
    """
    assert a.shape == b.shape
    C = np.zeros((np.size(unique_a), np.size(unique_b)))
    for i, la in enumerate(unique_a):
        for j, lb in enumerate(unique_b):
            C[i, j] = np.sum(np.logical_and(a == la, b == lb))
    return C


def matrix_scatterplot(M, xlabels, ylabels, fig_width=10, fig_height=14, scale_factor=10.0):
    """Plots a matrix with points as in a scatterplot. Area of points proportional to each matrix element. 
    Suitable to show sparse matrices.

    Args:
        M (np.array): a 2D array
        xlabels: label list
        ylabels: label list
        fig_width (int): matplotlib figure width
        fig_height (int): matplotlib figure height
        scale_factor (float): scales the points by this value. 
    """
    Mplot = M.copy()*scale_factor
    Mplot = np.flip(Mplot, axis=0)
    ylabels.reverse()
    x = np.arange(0, M.shape[1], 1)
    y = np.arange(0, M.shape[0], 1)
    xx, yy = np.meshgrid(x, y)
    plt.figure(figsize=(fig_width, fig_height))
    plt.scatter(np.ravel(xx), np.ravel(yy), s=np.ravel(Mplot), c='dodgerblue')
    ax = plt.gca()
    ax.set_xlim(np.min(x)-0.5, np.max(x)+0.5)
    ax.set_ylim(np.min(y)-0.5, np.max(y)+0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, rotation=0)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')

    ax.tick_params(color='None')
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for tick in ax.get_yticklabels():
        #tick.set_fontname("DejaVu Sans Mono")
        tick.set_fontfamily('monospace')
        tick.set_fontsize(12)

    for tick in ax.get_xticklabels():
        tick.set_fontfamily('monospace')
        tick.set_fontsize(12)

    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
    plt.box(False)
    plt.tight_layout()
    plt.show()
    return
