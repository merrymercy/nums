import numpy as np
import matplotlib.pyplot as plt

method_order_list = [
    'Numpy', 'Cupy', 'CupyParallelSystem',
]

def method2order(method):
    return method_order_list.index(method)

def method2color(method):
    return "C%d" % method_order_list.index(method)

def show_name(method):
    return method

def draw_grouped_bar_chart(data, baseline=None, output='out.png',
        yscale_log=False, yticks=None, y_max=None,
        legend_bbox_to_anchor=None, legend_nrow=None,
        figure_size=None, figax=None, draw_ylabel=True, draw_legend=True,
        data_error_bar=None, title=None):
    """
    Parameters
    data: OrderedDict[workload_name -> OrderedDict[method] -> cost]]
    """
    width = 1
    gap = 1.5
    fontsize = 19
    xticks_font_size = fontsize - 2

    figure_size = figure_size or (11, 4)
    legend_bbox_to_anchor = legend_bbox_to_anchor or (0.45, 1.25)

    all_methods = set()
    legend_set = {}

    if figax is None:
        fig, ax = plt.subplots()
        axes = []
        axes.append(ax)
    else:
        # for drawing subplot
        ax = figax

    x0 = 0
    xticks = []
    xlabels = []

    workloads = list(data.keys())
    for wkl in workloads:
        ys = []
        colors = []

        methods = list(data[wkl].keys())

        if baseline in data[wkl]:
            baseline_cost = data[wkl][baseline]
        else:
            # normalize to best library
            baseline_cost = 1
            #for method in methods:
            #    if data[wkl][method] < baseline_cost:
            #        baseline_cost = data[wkl][method]

        methods.sort(key=lambda x: method2order(x))
        for method in methods:
            relative_speedup = data[wkl][method] / baseline_cost
            if yticks is None:
                ys.append(relative_speedup)
            else:
                ys.append(max(relative_speedup, yticks[0] * 1.1))
            colors.append(method2color(method))

        # draw the bars
        xs = np.arange(x0, x0 + len(ys))

        if data_error_bar:
            yerrs = [data_error_bar[wkl][method] for method in methods]
            bars = ax.bar(xs, ys, yerr=yerrs, width=width, color=colors, ecolor='dimgray')
        else:
            bars = ax.bar(xs, ys, width=width, color=colors)

        for method, bar_obj in zip(methods, bars):
            all_methods.add(method)
            if method not in legend_set:
                legend_set[method] = bar_obj

        # tick and label
        x0 += len(ys) + gap

        xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
        xlabels.append(show_name(wkl))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=xticks_font_size)
        plt.tick_params(axis='x', which='both', bottom='off', top='off')

        if draw_ylabel is True:
            ax.set_ylabel('Time Cost (s)', fontsize=fontsize)
        elif isinstance(draw_ylabel, str):
            ax.set_ylabel(draw_ylabel, fontsize=fontsize)

        if yscale_log:
            ax.set_yscale('log', basey=2)
        if yticks is not None:
            ax.set_yticks(yticks)
        if y_max:
            ax.set_ylim(top=y_max)

        from matplotlib.ticker import FormatStrFormatter
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.grid(linewidth=0.4, linestyle='dotted') # draw grid line
        ax.set_axisbelow(True)  # grid lines are behind the rest
        ax.tick_params(bottom=False, top=False, right=False)

    # put legend outside the plot
    all_methods = list(all_methods)
    all_methods.sort(key=lambda x : method2order(x))

    ax.set_xlabel("Dataset Size", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize, pad=60.0)

    if draw_legend:
        legend_nrow = legend_nrow or 2
        ncol = (len(all_methods) + legend_nrow - 1)// legend_nrow
        ax.legend([legend_set[x] for x in all_methods],
                  [show_name(x) for x in all_methods],
                  fontsize=fontsize-1,
                  loc='upper center',
                  bbox_to_anchor=legend_bbox_to_anchor,
                  ncol=ncol,
                  handlelength=1.0,
                  handletextpad=0.5,
                  columnspacing=1.1)

    if figax is None:
        fig.set_size_inches(figure_size)
        fig.savefig(output, bbox_inches='tight')
        print("Output the plot to %s" % output)


def read_data(in_file):
    data = {}

    for line in open(in_file):
        items = [x.strip() for x in line.split(",")]
        library, N, cost, cv = items
        N, cost, cv = [eval(x) for x in [N, cost, cv]]

        workload_name = "%.1f GB" % (N * 1000 * 4 / 1e9)

        if cost < 0:
            continue

        if workload_name not in data:
            data[workload_name] = {}

        data[workload_name][library] = cost

    return data


if __name__ == "__main__":
    data = read_data("result_bop.csv")
    draw_grouped_bar_chart(data, legend_nrow=1, title="Time cost for $X^TX$", yscale_log=True)

