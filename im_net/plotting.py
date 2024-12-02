import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib import legend_handler as lh

class HandlerTupleVertical(HandlerTuple):
    def __init__(self, **kwargs):
        HandlerTuple.__init__(self, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # How many lines are there.
        numlines = len(orig_handle)
        handler_map = legend.get_legend_handler_map()

        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        height_y = (height / numlines)

        leglines = []
        for i, handle in enumerate(orig_handle):
            handler = legend.get_legend_handler(handler_map, handle)

            legline = handler.create_artists(legend, handle,
                                             xdescent,
                                             (2*i + 1)*height_y,
                                             width,
                                             2*height,
                                             fontsize, trans)
            leglines.extend(legline)

        return leglines

def format_figure_broken_axis(ax, max_exp=2, spine='both', offset=False):

    ax.set_xscale('symlog', linthresh=1, linscale=.6)

    ax.set_xlim(0, 10**max_exp)

    # Broken axis
    d = 0.01
    broken_x = 0.07
    breakspacing = 0.015
    if offset==True:
        offset = 0.15
    elif offset == False:
        offset = 0
    else:
        offset = offset
    
    kwargs = dict(transform=ax.transAxes, color='k',
                  clip_on=False, linewidth=.8, zorder=4)
    if spine in ['top', 'both']:
        ax.plot((broken_x-breakspacing*0.9, broken_x+breakspacing*0.9), (1, 1),
            color='w', transform=ax.transAxes, clip_on=False, linewidth=.8, zorder=3)
        ax.plot((broken_x-d-breakspacing, broken_x+d - breakspacing), (1-3*d, 1+3*d), **kwargs)
        ax.plot((broken_x-d+breakspacing, broken_x+d + breakspacing), (1-3*d, 1+3*d), **kwargs)
    if spine in ['bottom', 'both']:
        ax.plot((broken_x-breakspacing*0.9, broken_x+breakspacing*0.9), (0-offset, 0-offset),
            color='w', transform=ax.transAxes, clip_on=False, linewidth=.8, zorder=3)
        ax.plot((broken_x-d-breakspacing, broken_x+d - breakspacing), (-3*d-offset, +3*d-offset), **kwargs)
        ax.plot((broken_x-d+breakspacing, broken_x+d + breakspacing), (-3*d-offset, +3*d-offset), **kwargs)
#     
def change_log_ticks(ax, axis="x"):
    from matplotlib.ticker import ScalarFormatter
    if axis == "x":
        ax.xaxis.set_major_formatter(ScalarFormatter())
    elif axis == "y":
        ax.yaxis.set_major_formatter(ScalarFormatter())
    elif axis == "both":
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())

def plot_min_max(ax, x, data, midpoint, label=None, ls='-', **kwargs):
    assert midpoint in ["mean", "median"]
    maxs = np.zeros(data.shape[0])
    mins = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        maxs[i] = np.max(data[i][data[i] != 1])
        mins[i] = np.min(data[i][data[i] != 1])

    ax.scatter(x, maxs, marker='_', **kwargs)
    ax.scatter(x, mins, marker='_', **kwargs)
    ax.vlines(x, mins, maxs, linestyle='-', **kwargs)

    if midpoint == "mean":
        return ax.plot(x, data.mean(axis=1), label=label, linestyle=ls, **kwargs)
    elif midpoint == "median":
        return ax.plot(x, np.median(data, axis=1), label=label, linestyle=ls, **kwargs)




def add_to_legend(legend, label, color, **kwargs):
    ax = legend.axes
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color=color, **kwargs))
    labels.append(label)

    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())


def plot_batch_data(ax, data, x_range, **kwargs):  # Find better name
    data_len = data.shape[0]
    x = np.arange(data_len) / data_len * x_range
    data_trimmed = np.trim_zeros(data, "b")
    x = x[: data_trimmed.shape[0]]
    ax.plot(x, data_trimmed, **kwargs)


def adjust_spines(ax, spines, offset=10, offset_spines='all'):
    if offset_spines == 'all':
        offset_spines = spines

    for loc, spine in ax.spines.items():
        if loc in offset_spines:
            spine.set_position(("outward", offset))  # outward by 10 points
        elif loc in spines:
            spine.set_position(("outward", 0))  # outward by 0 points
        else:
            spine.set_color("none")  # don't draw spine

    # turn off ticks where there is no spine
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def check_ax_and_colors(ax, colors, num_plots):
    if ax is None:
        ax = plt.gca()
    if colors is None or len(colors) != num_plots:
        colors = sns.color_palette("colorblind", num_plots)
    return ax, colors

def get_atom_labels(num_atoms):
    if num_atoms == 5:
        return ['(1)', '(2)', '(1)(2)','(12)','h_res']
    elif num_atoms == 19:
        return ["(1)(2)(3)","(1)(2)","(1)(3)","(2)(3)","(1)(23)","(2)(13)","(3)(12)","(1)","(2)","(3)","(12)(13)(23)","(12)(13)","(12)(23)","(13)(23)","(12)","(13)","(23)","(123)","h_res"]
    else:
        print('Invalid number of atoms')
        return None

def plot_information_terms(
    ax,
    atom_data,
    labels=None,
    colors=None,
    legend=True,
    std=None,
    set_labels=False,
    max=None,
    min=None,
    **kwargs,
):
    ax, colors = check_ax_and_colors(ax, colors, len(atom_data.T))

    if labels is None:
        labels = get_atom_labels(len(atom_data[0]))

    for i, atom in enumerate(atom_data.T):
        ax.plot(atom, label=labels[i], color=colors[i], **kwargs)
        if std is not None:
            ax.fill_between(
                np.arange(len(atom)),
                atom - std[:, i],
                atom + std[:, i],
                alpha=0.2,
                color=colors[i],
            )

        if max is not None and min is not None:
            ax.fill_between(
                np.arange(len(atom)), max[:, i], min[:, i], alpha=0.2, color=colors[i]
            )

    if set_labels:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Information in bits")

    if legend:
        ax.legend()

def plot_performance_list(ax, data, colors=None, linestyle='-', **kwargs):
    pass

def plot_pid_atom_list(ax, pid_atoms, layer, colors, labels=None, percentile=10, clip_on=False, include_sum=False, **kwargs):
    num_atoms = pid_atoms[0][layer].shape[1]
    num_runs = len(pid_atoms)
    ax, colors = check_ax_and_colors(ax, colors, num_atoms)
    if labels is None:
        labels = get_atom_labels(num_atoms)
    atoms = np.empty((num_atoms,0)).tolist()
    for run_idx in range(num_runs):
        for atom_idx in range(num_atoms):
            atoms[atom_idx].append(pid_atoms[run_idx][layer][:,atom_idx])
    
    for atom_idx, atom in enumerate(atoms):
        atom = np.array(atom)
        atom = np.swapaxes(atom, 1, 2)
        num_datapoints = atom.shape[1]
        atom = np.reshape(atom, (-1,101))
        atoms[atom_idx] = atom
        if colors[atom_idx] == 'black':
            alpha = 0.02
        else:
            alpha = 1
        ax.plot(np.median(atom,axis=0), label=labels[atom_idx], color=colors[atom_idx], alpha=alpha,clip_on=clip_on)
        if percentile is not None:
            ax.fill_between(range(101), np.percentile(atom, percentile, axis=0), np.percentile(atom, 100-percentile, axis=0), alpha=0.1, clip_on=clip_on, color=colors[atom_idx])

    if include_sum:
        sum = np.sum(atoms, axis=0)
        ax.plot(np.median(sum, axis=0), label="H(T)", color='black', clip_on=clip_on)
        if percentile is not None:
            ax.fill_between(range(101), np.percentile(sum, percentile, axis=0), np.percentile(sum, 100-percentile, axis=0), alpha=0.1, clip_on=clip_on, color='black')

    # for atom_idx in range(num_atoms):
    #     atoms[atom_idx] = np.array(atoms[atom_idx])
    #     num_datapoints = atoms[atom_idx].shape[1]
    #     ax.plot(atoms[atom_idx].mean(axis=0).mean(axis=1), label=labels[atom_idx], color=colors[atom_idx], clip_on=clip_on)
    #     if std:
    #         ax.fill_between(range(num_datapoints), atoms[atom_idx].mean(axis=0).mean(axis=1)-atoms[atom_idx].std(axis=0).mean(axis=1), atoms[atom_idx].mean(axis=0).mean(axis=1)+atoms[atom_idx].std(axis=0).mean(axis=1), alpha=0.1, clip_on=clip_on, color=colors[atom_idx])
    #     elif minmax:
    #         ax.fill_between(range(num_datapoints), atoms[atom_idx].min(axis=0).min(axis=1), atoms[atom_idx].max(axis=0).max(axis=1), alpha=0.1, clip_on=clip_on, color=colors[atom_idx])
    #     elif percentile != False:
    #         ax.fill_between(range(num_datapoints), np.percentile(atoms[atom_idx], percentile[0], axis=0).mean(axis=1), np.percentile(atoms[atom_idx], percentile[1], axis=0).mean(axis=1), alpha=0.1, clip_on=clip_on, color=colors[atom_idx])
    # if include_sum:
    #     sum = np.sum(atoms, axis=0)
    #     ax.plot(sum.mean(axis=0).mean(axis=1), label="H(T)", color='black', clip_on=clip_on)
    #     if std:
    #         ax.fill_between(range(num_datapoints), sum.mean(axis=0).mean(axis=1)-sum.std(axis=0).mean(axis=1), sum.mean(axis=0).mean(axis=1)+sum.std(axis=0).mean(axis=1), alpha=0.1, clip_on=clip_on, color='black')
    #     elif minmax:
    #         ax.fill_between(range(num_datapoints), sum.min(axis=0).min(axis=1), sum.max(axis=0).max(axis=1), alpha=0.1, clip_on=clip_on, color='black')
    #     elif percentile != False:
    #         ax.fill_between(range(num_datapoints), np.percentile(sum, percentile[0], axis=0).mean(axis=1), np.percentile(sum, percentile[1], axis=0).mean(axis=1), alpha=0.1, clip_on=clip_on, color='black')


def plot_performance(
    ax, performance_data, colors=None, legend=True, ax2=None, **kwargs
):
    """Expects a dictionary of performance data with keys as labels and values as lists of performance values."""
    ax, colors = check_ax_and_colors(ax, colors, len(performance_data))
    ax.plot(
        performance_data["train_loss"],
        color=colors[0],
        ls="--",
        label="Train Loss",
        zorder=2,
        **kwargs,
    )
    ax.plot(
        performance_data["val_loss"],
        color=colors[1],
        ls="--",
        label="Test Loss",
        zorder=2,
        **kwargs,
    )

    if ax2 is None:
        ax2 = ax.twinx()

    ax2.plot(
        performance_data["train_acc"],
        color=colors[0],
        label="Train Acc",
        zorder=3,
        **kwargs,
    )
    ax2.plot(
        performance_data["val_acc"],
        color=colors[1],
        label="Test Acc",
        zorder=3,
        **kwargs,
    )
    # set limit of right axis

    ax2.set_ylim(-0.1, 1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    if legend:
        ax.legend()
        ax2.legend()

    return ax, ax2


def plot_weight_fields(layer, axes, shapes=[(28, 28), (10, 1)], **kwargs):
    for i in range(layer.sources[0].weight.shape[0]):
        if i == 0:
            for s in range(len(layer.sources)):
                axes[s, i].set_ylabel(f"Input weights {layer.source_names[s]}")
        for s in range(len(layer.sources)):
            axes[s, i].imshow(
                layer.sources[s].weight[i].view(*shapes[s]).detach().cpu().numpy(),
                **kwargs,
            )
    remove_all_ticks(axes)


def plot_hist(hist, axes, **kwargs):
    for i in range(axes.shape[0]):
        axes[i].imshow(hist[i], **kwargs)


def remove_all_ticks(ax):
    ax2 = ax.flatten().tolist()
    for a in ax2:
        a.set_xticks([])
        a.set_yticks([])


def plot_xy_with_noise(ax, x, y, x_label=None, y_label=None, colors=None, **kwargs):
    ax, colors = check_ax_and_colors(ax, colors, 1)
    x = np.array(x.flatten())
    y = np.array(y.flatten())
    px = np.random.randn(len(x)) * 0.1
    py = np.random.randn(len(y)) * 0.1
    ax.plot(x + px, y + py, "x", **kwargs)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    plt.tight_layout()


def postProcess(dm, which="all"):
    os.makedirs(os.path.join(dm.exp_directory, "Plots"), exist_ok=True)
    if which == "all" or which == "weights":
        perf = dm.load_data_rec("performance")
        fig, ax = plt.subplots(1, 1)
        plot_performance(ax, perf, legend=True)
        fig.savefig(os.path.join(dm.exp_directory, "Plots", "performance.pdf"))
        plt.close(fig)

    if which == "all" or which == "info_quantities":
        info_quantities = dm.load_data_rec("info_quantities")
        for layer in info_quantities:
            os.makedirs(os.path.join(dm.exp_directory, "Plots", layer), exist_ok=True)
            fig, ax = plt.subplots(1, 1)
            # TODO: make the plotting work for 10 neurons, even if there are more
            if len(info_quantities[layer]) < 11:
                for neuron, info_terms in info_quantities[layer].items():
                    labels = get_atom_labels(len(info_terms[0]))
                    plot_information_terms(ax, info_terms, labels=labels, legend=True)
                    ax.set_title(f"Neuron {neuron}")
                    fig.savefig(
                        os.path.join(dm.exp_directory, "Plots", layer, f"{neuron}.pdf")
                    )
                    ax.clear()
                plt.close(fig)


def plot_2D_decision_boundaries(layer, axes, x_range, y_range, **kwargs):
    import torch

    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    # construct X[0]
    x0 = torch.Tensor(np.stack([X.flatten(), Y.flatten()], axis=1))
    x1 = torch.zeros(x0.shape[0], layer.output_size)
    Z = layer.forward([x0, x1], sample=False).detach().numpy()
    for n in range(Z.shape[1]):
        for i in range(Z.shape[2]):
            axes[n, i].contourf(X, Y, Z[:, n, i].reshape(X.shape), **kwargs)
