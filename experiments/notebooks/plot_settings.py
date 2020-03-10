import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec, cm

import palettable

TEXTWIDTH = 7.1014  # inches

# CMAP = palettable.scientific.sequential.Batlow_20.mpl_colormap
# CMAP = palettable.scientific.sequential.Tokyo_20.mpl_colormap
# CMAP = palettable.scientific.sequential.LaJolla_20_r.mpl_colormap
CMAP = cm.plasma
CMAP_R = cm.plasma_r

# COLORS = palettable.scientific.sequential.Batlow_6.mpl_colors
# COLORS = palettable.scientific.sequential.Tokyo_5.mpl_colors
COLORS = [CMAP(i/4.) for i in range(5)]

COLOR_EF = COLORS[0]
COLOR_PIE = COLORS[1]
COLOR_MLFS = COLORS[2]
COLOR_MLFA = COLORS[3]
COLOR_MLFOT = COLORS[4]
COLOR_EMLFS = COLORS[2]
COLOR_EMLFA = COLORS[3]

COLOR_NEUTRAL1 = "black"
COLOR_NEUTRAL2 = "0.7"
COLOR_NEUTRAL3 = "0.4"


def setup():
    matplotlib.rcParams.update({'font.size': 10})
    # matplotlib.rcParams.update({'text.usetex': True, 'font.size': 10, 'font.family': 'serif'})
    # params= {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
    # plt.rcParams.update(params)


def figure(cbar=False, height=TEXTWIDTH*0.5, large_margin=0.18, mid_margin=0.14, small_margin=0.03, cbar_sep=0.02, cbar_width=0.04):
    if cbar:
        width = height * (1. + cbar_sep + cbar_width + large_margin - small_margin)
        top = small_margin
        bottom = mid_margin
        left = large_margin
        right = large_margin + cbar_width + cbar_sep
        cleft = 1. - (large_margin + cbar_width) * height / width
        cbottom = bottom
        cwidth = cbar_width * height / width
        cheight = 1. - top - bottom

        fig = plt.figure(figsize=(width, height))
        ax = plt.gca()
        plt.subplots_adjust(
            left=left * height / width,
            right=1. - right * height / width,
            bottom=bottom,
            top=1. - top,
            wspace=0.,
            hspace=0.,
        )
        cax = fig.add_axes([cleft, cbottom, cwidth, cheight])

        plt.sca(ax)

        return fig, (ax, cax)
    else:
        width = height
        left = mid_margin
        right = small_margin
        top = small_margin
        bottom = mid_margin

        fig = plt.figure(figsize=(width, height))
        ax = plt.gca()
        plt.subplots_adjust(
            left=left,
            right=1. - right,
            bottom=bottom,
            top=1. - top,
            wspace=0.,
            hspace=0.,
        )

        return fig, ax


def grid(nx=4, ny=2, height=0.5*TEXTWIDTH, large_margin=0.14, small_margin=0.03, sep=0.02):
    # Geometry
    left = large_margin
    right = small_margin
    top = small_margin
    bottom = large_margin

    panel_size = (1. - top - bottom - (ny - 1)*sep)/ny
    width = height*(left + nx*panel_size + (nx-1)*sep + right)

    # wspace and hspace are complicated beasts
    avg_width_abs = (height*panel_size * nx * ny) / (nx * ny + ny)
    avg_height_abs = height*panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx, width_ratios=[1.]*nx, height_ratios=[1.] * ny)
    plt.subplots_adjust(
        left=left * height / width,
        right=1. - right * height / width,
        bottom=bottom,
        top=1. - top,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs


def grid_width(nx=4, ny=2, width=TEXTWIDTH, large_margin=0.14, small_margin=0.03, sep=0.03):
    left = large_margin
    right = small_margin
    top = small_margin
    bottom = large_margin
    panel_size = (1. - top - bottom - (ny - 1)*sep)/ny
    height = width / (left + nx*panel_size + (nx - 1)*sep + right)
    return grid2(nx, ny, height, large_margin, small_margin, sep)


def grid2(nx=4, ny=2, height=6., large_margin=0.14, small_margin=0.03, sep=0.03, cbar_width=0.06):
    # Geometry
    left = small_margin
    right = large_margin
    top = small_margin
    bottom = small_margin

    panel_size = (1. - top - bottom - (ny - 1)*sep)/ny
    width = height*(left + nx*panel_size + cbar_width + nx*sep + right)

    # wspace and hspace are complicated beasts
    avg_width_abs = (height*panel_size * nx * ny + ny * cbar_width * height) / (nx * ny + ny)
    avg_height_abs = height*panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(ny, nx + 1, width_ratios=[1.]*nx + [cbar_width], height_ratios=[1.] * ny)
    plt.subplots_adjust(
        left=left * height / width,
        right=1. - right * height / width,
        bottom=bottom,
        top=1. - top,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, gs


def grid2_width(nx=4, ny=2, width=TEXTWIDTH, large_margin=0.14, small_margin=0.03, sep=0.03, cbar_width=0.06):
    left = small_margin
    right = small_margin
    top = small_margin
    bottom = small_margin
    panel_size = (1. - top - bottom - (ny - 1)*sep)/ny
    height = width / (left + nx*panel_size + cbar_width + nx*sep + right)
    return grid2(nx, ny, height, large_margin, small_margin, sep, cbar_width)


def two_figures(height=TEXTWIDTH*0.4,  large_margin=0.18, small_margin=0.05, sep=0.21,):
    # Geometry (in multiples of height)
    left = large_margin
    right = small_margin
    top = small_margin
    bottom = large_margin
    panel_size = 1. - top - bottom

    # Absolute width
    width = height*(left + 2*panel_size+ sep + right)

    # wspace and hspace are complicated beasts
    avg_width_abs = height*panel_size
    avg_height_abs = height*panel_size
    wspace = sep * height / avg_width_abs
    hspace = sep * height / avg_height_abs

    # Set up figure
    fig = plt.figure(figsize=(width, height))
    plt.subplots_adjust(
        left=left * height / width,
        right=1. - right * height / width,
        bottom=bottom,
        top=1. - top,
        wspace=wspace,
        hspace=hspace,
    )

    ax_left = plt.subplot(1,2,1)
    ax_right = plt.subplot(1,2,2)

    return fig, ax_left, ax_right



# def grid(nx=4, ny=2, height=6., n_caxes=0, large_margin=0.02, small_margin=0.02, sep=0.02, cbar_width=0.03):
#     # Geometry (in multiples of height)
#     left = large_margin
#     right = small_margin
#     top = small_margin
#     bottom = large_margin
#     panel_size = (1. - top - bottom - (ny - 1)*sep)/ny
#
#     # Absolute width
#     width = height*(left + nx*panel_size+ (nx-1)*sep + right)
#
#     # wspace and hspace are complicated beasts
#     avg_width_abs = (height*panel_size * nx * ny + n_caxes * cbar_width * height) / (nx * ny + n_caxes)
#     avg_height_abs = height*panel_size
#     wspace = sep * height / avg_width_abs
#     hspace = sep * height / avg_height_abs
#
#     # Set up figure
#     fig = plt.figure(figsize=(width, height))
#     plt.subplots_adjust(
#         left=left * height / width,
#         right=1. - right * height / width,
#         bottom=bottom,
#         top=1. - top,
#         wspace=wspace,
#         hspace=hspace,
#     )
#
#     # Colorbar axes in last panel
#     caxes = []
#     if n_caxes > 0:
#         ax = plt.subplot(ny, nx, nx*ny)
#         ax.axis("off")
#         pos = ax.get_position()
#         cax_total_width=pos.width / n_caxes
#         cbar_width_ = cbar_width * height / width
#         for i in range(n_caxes):
#             cax = fig.add_axes([pos.x0 + i * cax_total_width, pos.y0, cbar_width_, pos.height])
#             cax.yaxis.set_ticks_position('right')
#             caxes.append(cax)
#
#     return fig, caxes
#
#
# def grid_width(nx=4, ny=2, width=TEXTWIDTH, n_caxes=0, large_margin=0.025, small_margin=0.025, sep=0.025, cbar_width=0.04):
#     left = large_margin
#     right = small_margin
#     top = small_margin
#     bottom = large_margin
#     panel_size = (1. - top - bottom - (ny - 1) * sep) / ny
#     height = width / (left + nx * panel_size + (nx - 1) * sep + right)
#     return grid(nx, ny, height, n_caxes, large_margin, small_margin, sep, cbar_width)