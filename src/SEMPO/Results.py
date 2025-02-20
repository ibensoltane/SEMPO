import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def GetAxCoordinates(ax):
    fig = ax.figure
    figH = fig.bbox.height
    figW = fig.bbox.width

    bbox = np.array([ax.bbox.x0 / figW, ax.bbox.y0 / figH, ax.bbox.width / figW, ax.bbox.height / figH])

    return bbox


def AmplitudeMap(Hw, W, ax=None, figID=1, fs=(5, 4), adjustFigure=False):
    cMap = np.array(sns.color_palette("Spectral_r", 15).as_hex())
    cMap = ListedColormap(cMap)

    cbar_ax_coord = None
    if ax is None:
        fig = plt.figure(figID, figsize=fs)
        plt.clf()
        ax = fig.subplots(1)
        ax_coord = GetAxCoordinates(ax)
        axRight = ax_coord[0] + ax_coord[2]
        if adjustFigure:
            axRight = 0.83
            fig.subplots_adjust(left=0.17, right=axRight, bottom=0.17, top=0.99)
        cbar_ax_coord = GetAxCoordinates(ax)
        cbar_ax_coord[0] = axRight + 0.013
        cbar_ax_coord[2] = min(0.04, 1 - axRight - 0.05)
    else:
        fig = ax.figure
        ax_coord = GetAxCoordinates(ax)
        axRight = ax_coord[0] + ax_coord[2]
        if adjustFigure:
            axRight = 0.83
            fig.subplots_adjust(left=0.17, right=axRight, bottom=0.17, top=0.99)
        cbar_ax_coord = GetAxCoordinates(ax)
        cbar_ax_coord[0] = axRight + 0.013
        cbar_ax_coord[2] = min(0.04, 1 - axRight - 0.05)

    a = ax.imshow(np.real(np.log(Hw)), extent=[np.min(np.real(W)), np.max(np.real(W)),
                                               np.min(np.imag(W)), np.max(np.imag(W))],
                  origin="lower", aspect="auto", cmap=cMap)
    cbar_ax = fig.add_axes(cbar_ax_coord)
    fig.colorbar(a, cax=cbar_ax, orientation='vertical')

    ax.set_xlabel(r"Re[$\omega$] ($10^{15}$ rad/s)", fontsize=11)
    ax.set_xlim(np.min(np.real(W)), np.max(np.real(W)))
    ax.set_ylabel(r"Im[$\omega$] ($10^{15}$ rad/s)", fontsize=11)
    ax.set_ylim(np.min(np.imag(W)[:, 0]), np.max(np.imag(W)[:, 0]))

    return ax.figure, ax


def AmplitudeCurve(Hw, W, ax=None, figID=1, fs=(5, 4)):
    if ax is None:
        fig = plt.figure(figID, figsize=fs)
        plt.clf()
        ax = fig.subplots(1)
    else:
        fig = ax.figure

    ax.plot(W, np.abs(Hw), color="black")
    ax.set_xlabel(r"$\omega$ ($10^{15}$ rad/s)", fontsize=11)
    ax.set_xlim(W[0], W[-1])
    ax.set_ylabel("Amplitude", fontsize=11)

    return ax.figure, ax


def PhaseMap(Hw, W, ax=None, figID=1, fs=(5, 4), adjustFigure=False):
    cMap = np.array(sns.color_palette("YlGnBu_r", 15).as_hex())
    cMap = ListedColormap(cMap)

    cbar_ax_coord = None
    if ax is None:
        fig = plt.figure(figID, figsize=fs)
        plt.clf()
        ax = fig.subplots(1)
        ax_coord = GetAxCoordinates(ax)
        axRight = ax_coord[0] + ax_coord[2]
        if adjustFigure:
            axRight = 0.83
            fig.subplots_adjust(left=0.16, right=axRight, bottom=0.15, top=0.97)
        cbar_ax_coord = GetAxCoordinates(ax)
        cbar_ax_coord[0] = axRight + 0.013
        cbar_ax_coord[2] = min(0.04, 1 - axRight - 0.05)
    else:
        fig = ax.figure
        ax_coord = GetAxCoordinates(ax)
        axRight = ax_coord[0] + ax_coord[2]
        if adjustFigure:
            axRight = 0.83
            fig.subplots_adjust(left=0.16, right=axRight, bottom=0.15, top=0.97)
        cbar_ax_coord = GetAxCoordinates(ax)
        cbar_ax_coord[0] = axRight + 0.013
        cbar_ax_coord[2] = min(0.04, 1 - axRight - 0.05)

    a = ax.imshow(np.imag(np.log(Hw)) * 180 / np.pi, extent=[np.min(np.real(W)), np.max(np.real(W)),
                                                             np.min(np.imag(W)), np.max(np.imag(W))],
                  origin="lower", aspect="auto", cmap=cMap)
    cbar_ax = fig.add_axes(cbar_ax_coord)
    fig.colorbar(a, cax=cbar_ax, orientation='vertical')

    ax.set_xlabel(r"Re[$\omega$] ($10^{15}$ rad/s)", fontsize=11)
    ax.set_xlim(np.min(np.real(W)[0, :]), np.max(np.real(W)[0, :]))
    ax.set_ylabel(r"Im[$\omega$] ($10^{15}$ rad/s)", fontsize=11)
    ax.set_ylim(np.min(np.imag(W)[:, 0]), np.max(np.imag(W)[:, 0]))

    return ax.figure, ax


def PhaseCurve(Hw, W, ax=None, figID=1, fs=(5, 4)):
    if ax is None:
        fig = plt.figure(figID, figsize=fs)
        plt.clf()
        ax = fig.subplots(1)
    else:
        fig = ax.figure

    ax.plot(W, np.unwrap(np.imag(np.log(Hw))) * 180 / np.pi, color="black")
    ax.set_xlabel(r"$\omega$ ($10^{15}$ rad/s)", fontsize=11)
    ax.set_xlim(W[0], W[-1])
    ax.set_ylabel(r"Phase ($\degree$)", fontsize=11)

    return ax.figure, ax


def BodeDiagram(Hw, W, figID=1, fs=(5, 4)):
    fig = plt.figure(figID, figsize=fs)
    plt.clf()
    axs = fig.subplots(2, 1)

    fig, ax1 = AmplitudeCurve(Hw, W, ax=axs[0])
    ax1.set_xlabel("")
    fig, ax2 = PhaseCurve(Hw, W, ax=axs[1])

    return fig


def BodeMaps(Hw, W, figID=1, fs=(5, 4)):
    fig = plt.figure(figID, figsize=fs)
    plt.clf()
    axs = fig.subplots(2, 1)

    fig, ax1 = AmplitudeMap(Hw, W, ax=axs[0], adjustFigure=True)
    ax1.set_xlabel("")
    fig, ax2 = PhaseMap(Hw, W, ax=axs[1], adjustFigure=False)

    return fig


def AmplitudeFig(Hx, Wx, Hw, W, figID=1, fs=(5, 4)):
    fig = plt.figure(figID, figsize=fs)
    plt.clf()
    axs = fig.subplots(2, 1)

    fig, ax1 = AmplitudeCurve(Hw, W, ax=axs[0])
    ax1.set_xlabel("")
    ax1.set_xlim(min(np.min(np.real(Wx)), W[0]), min(np.max(np.real(Wx)), W[-1]))
    fig, ax2 = AmplitudeMap(Hx, Wx, ax=axs[1], adjustFigure=True)
    ax2.set_xlim(min(np.min(np.real(Wx)), W[0]), min(np.max(np.real(Wx)), W[-1]))

    return fig


def PhaseFig(Hx, Wx, Hw, W, figID=1, fs=(5, 4)):
    fig = plt.figure(figID, figsize=fs)
    plt.clf()
    axs = fig.subplots(2, 1)

    fig, ax1 = PhaseCurve(Hw, W, ax=axs[0])
    ax1.set_xlabel("")
    ax1.set_xlim(min(np.min(np.real(Wx)), W[0]), min(np.max(np.real(Wx)), W[-1]))
    fig, ax2 = PhaseMap(Hx, Wx, ax=axs[1], adjustFigure=True)
    ax2.set_xlim(min(np.min(np.real(Wx)), W[0]), min(np.max(np.real(Wx)), W[-1]))

    return fig


def ParameterDistribution(poles, figID=1, fs=(5, 4), wr1=None, wr2=None, wi1=None, wi2=None, freq=True):
    fig = plt.figure(figID, figsize=fs)
    plt.clf()
    ax = fig.subplots(1)
    fig.subplots_adjust(left=0.16, bottom=0.15)

    ax.scatter(x=np.real(poles), y=np.imag(poles), s=12, color="red", zorder=200)

    if freq:
        ax.set_xlabel(r"Re[$\omega$] ($10^{15}$ rad/s)", fontsize=11)
        ax.set_ylabel(r"Im[$\omega$] ($10^{15}$ rad/s)", fontsize=11)

    wr1 = np.min(np.real(poles)) - 0.1 if wr1 is None else wr1
    wr2 = np.max(np.real(poles)) + 0.1 if wr2 is None else wr2
    wi1 = np.min(np.imag(poles)) - 0.1 if wi1 is None else wi1
    wi2 = np.max(np.imag(poles)) + 0.1 if wi2 is None else wi2

    ax.set_xlim(wr1, wr2)
    ax.set_ylim(wi1, wi2)

    return fig, ax