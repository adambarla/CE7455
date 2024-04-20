import numpy as np
from matplotlib import pyplot as plt


def aggregate(df, param_tuples, result_column, metric="mean"):
    df_agg = df.groupby(param_tuples).agg({result_column: [metric]})
    return df_agg


def plot_heatmap_on_ax(
    ax,
    results,
    param1_values,
    param2_values,
    cmap,
    xlabels=False,
    ylabels=False,
    vmin=0,
    vmax=1,
    pos="NW",
):
    cax = ax.matshow(results, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_yticks(
        np.arange(len(param1_values)), param1_values if ylabels else [], fontsize=22
    )
    ax.set_xticks(
        np.arange(len(param2_values)),
        param2_values if xlabels else [],
        rotation="vertical",
        fontsize=22,
    )

    ax.tick_params(
        axis="both",
        right="E" in pos,
        left="W" in pos,
        bottom="S" in pos,
        top="N" in pos,
        labelbottom="S" in pos,
        labeltop="N" in pos,
        labelleft="W" in pos,
        labelright="E" in pos,
    )
    ax.grid(False)
    return cax


def calcualte_vs(df_runs, params, result_column, metric="mean", vmin=None, vmax=None):
    if vmin != None and vmax != None:
        return vmin, vmax
    values = []
    for param1 in params:
        for param2 in params:
            if param1 == param2:
                continue
            df_agg = aggregate(df_runs, [param1, param2], result_column, metric)[
                (result_column, metric)
            ].unstack()
            values.extend(df_agg.values.flatten())

    vmin = min(values) if vmin is None else vmin
    vmax = max(values) if vmax is None else vmax
    return vmin, vmax


# grid of (n-1 x n-1) plots, n is number of hyperparameters, each plot is a heatmap of mean test acc for different hyperparameter value combinations
def grid(
    df_runs,
    params,
    result_column,
    metric="mean",
    cmap="YlGn",
    vmin=None,
    vmax=None,
    figsize=(10, 10),
    rename_dict=None,
):
    n = len(params)
    fig, axs = plt.subplots(nrows=n - 1, ncols=n - 1, figsize=figsize)
    fig.tight_layout(pad=3.0)

    rename = lambda x: rename_dict[x] if rename_dict and x in rename_dict else x

    vmin, vmax = calcualte_vs(
        df_runs, params, result_column, metric, vmin=vmin, vmax=vmax
    )
    for i in range(n - 1):
        for j in range(1, i + 1):
            axs[i][j - 1].axis("off")
        for j in range(i + 1, n):
            param1 = params[i]
            param2 = params[j]
            ax = axs[i, j - 1]
            df_agg = aggregate(df_runs, [param1, param2], result_column, metric)[
                (result_column, metric)
            ].unstack()
            cax = plot_heatmap_on_ax(
                axs[i][j - 1],
                df_agg,
                [rename(x) for x in df_agg.index],
                [rename(x) for x in df_agg.columns],
                cmap,
                i == 0,
                j == n - 1,
                vmin=vmin,
                vmax=vmax,
                pos="NE",
            )

            # print labels
            if i == j - 1:
                # position lable on top
                axs[i][j - 1].set_xlabel(rename(param2), fontsize=22, fontweight="bold")
                axs[i][j - 1].xaxis.set_label_coords(0.5, -0.2)
                # position the label a bit down
                if j != n - 1:
                    axs[i][j - 1].xaxis.set_label_coords(0.5, -0.4)
                if i == 0:
                    axs[i][j - 1].set_ylabel(
                        rename(param1), fontsize=22, fontweight="bold", rotation=0
                    )
                    axs[i][j - 1].yaxis.set_label_coords(-0.4, 0.4)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    for ax in axs.flat:
        ax.set_anchor("NE")

    # colorbar anchor to the left
    ticks = np.linspace(vmin, vmax, 4)
    cb = fig.colorbar(
        cax,
        ax=axs[-1, 0:2],
        orientation="horizontal",
        aspect=7,
        ticks=ticks,
    )
    # # legend with red dot and best hyperparameters
    # cb.ax.legend([plt.scatter([], [], marker='o', color='red')], ['best hyperparameters'], loc='lower center', ncol=2,
    #              fontsize=16, frameon=False, bbox_to_anchor=(0.5, 1))
    # colorbar ticks
    cb.ax.set_xticklabels([f"{tick:.2f}" for tick in ticks], fontsize=22)
    return fig


def combined_grid(
    df1,
    df2,
    params,
    result_column,
    metric="mean",
    cmap1="YlGn",
    cmap2="YlGn",
    vmin=None,
    vmax=None,
    figsize=(10, 10),
    rename_dict=None,
):
    n = len(params)
    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=figsize)
    fig.tight_layout(pad=3.0)

    rename = lambda x: rename_dict[x] if rename_dict and x in rename_dict else x

    vmin1, vmax1 = calcualte_vs(
        df1, params, result_column, metric, vmin=vmin, vmax=vmax
    )
    vmin2, vmax2 = calcualte_vs(
        df2, params, result_column, metric, vmin=vmin, vmax=vmax
    )

    for i in range(n):
        # top part
        for j in range(i + 1, n):
            param1 = params[i]
            param2 = params[j]
            ax = axs[i][j]
            df_agg = aggregate(df1, [param1, param2], result_column, metric)[
                ("eval", "test/acc", metric)
            ].unstack()
            cax1 = plot_heatmap_on_ax(
                ax,
                df_agg,
                [rename(x) for x in df_agg.index],
                [rename(x) for x in df_agg.columns],
                cmap1,
                i == 0,
                j == 0,
                vmin=vmin1,
                vmax=vmax1,
            )

        # bottom part
        for j in range(0, i):
            ax = axs[i][j]
            param1 = params[i]
            param2 = params[j]
            df_agg = aggregate(df2, [param1, param2], result_column, metric)[
                (result_column, metric)
            ].unstack()
            cax2 = plot_heatmap_on_ax(
                ax,
                df_agg,
                [rename(x) for x in df_agg.index],
                [rename(x) for x in df_agg.columns],
                cmap2,
                i == 0,
                j == 0,
                vmin=vmin2,
                vmax=vmax2,
                pos="SW",
            )

        # labels
        ax = axs[i][i]
        ax.set_facecolor("white")
        param = params[i]
        df_agg1 = aggregate(df1, [param, param], metric)[
            (result_column, metric)
        ].unstack()
        df_agg2 = aggregate(df2, [param, param], metric)[
            (result_column, metric)
        ].unstack()
        param1_values = df_agg1.index
        param2_values = df_agg2.columns
        # mat  of nans of (df1 param i value count) x (df2 param i value count)
        mat = np.full((len(param1_values), len(param2_values)), np.nan)
        plot_heatmap_on_ax(
            ax,
            mat,
            [rename(x) for x in param1_values],
            [rename(x) for x in param2_values],
            cmap1,
            i == 0,
            i == 0,
            pos="NW",
        )
        ax.set_xlabel(
            rename(param[1]),
            fontsize=22,
            fontweight="bold",
            verticalalignment="center",
            horizontalalignment="center",
        )
        ax.xaxis.set_label_coords(0.5, 0.5)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    for ax in axs.flat:
        ax.set_anchor("NE")

    ticks1 = np.linspace(vmin1, vmax1, 5)
    # vertical on the right
    newax1 = fig.add_axes([0.75, 0, 0.3, 1], anchor="N")
    newax1.axis("off")
    cb1 = fig.colorbar(
        cax1,
        ax=newax1,
        orientation="vertical",
        aspect=10,
        ticks=ticks1,
    )
    cb1.ax.set_yticklabels([f"{tick:.0f}" for tick in ticks1], fontsize=22)

    # horizontal on the bottom
    ticks2 = np.linspace(vmin2, vmax2, 5)
    newax2 = fig.add_axes([0, -0.03, 1, 0.3], anchor="N")
    newax2.axis("off")
    cb2 = fig.colorbar(
        cax2, ax=newax2, orientation="horizontal", aspect=10, ticks=ticks2
    )
    cb2.ax.set_xticklabels([f"{tick:.0f}" for tick in ticks2], fontsize=22)

    return fig
