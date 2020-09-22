from matplotlib import pyplot as plt, colors
import utils
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx

from tqdm import tqdm, trange
from lsh import makeLSHIndex


def presentGraphsWithGEDs(self, entireDataset, withPrediction=False):
    '''
    This function presents pair of graphs that are both under 6 nodes and their GEDs. TODO more notes.
    :param entireDataset: Boolean value that determines whether the entire dataset should be presented or a sample
    :return:
    '''
    # Empty target folder from previous images
    if entireDataset:
        path = '../dataset_printed'
    else:
        path = '../comparative_plots'
    utils.clearDataFolder(path)

    # The node number per graph
    graphs_size = 6
    # num_of_figs determines how many pairs the sample contains
    if entireDataset is False:
        num_of_figs = 4
    # if you don't want to present num_of_figs pairs of each training graph change samplePerTrainGraph to False
    samplePerTrainGraph = False

    # DEBUGGING, check validity of ged, denormalization etc.
    # self.denorm_ground_truth, self.data_ged_part, self.denorm_prediction_mat = utils.verifyGedNormalization(self)
    # x = self.training_graphs.__len__()

    aids = True if self.args.dataset == 'AIDS700nef' else False
    # Create new images
    for j in tqdm(range(0, self.training_graphs.__len__() - 1)):
        g1_geom = self.training_graphs[j]
        g1_name = "train-{}".format(j)
        g1 = utils.PyG2NetworkxG(g1_geom, aids=aids)
        if g1.number_of_nodes() <= graphs_size or self.args.dataset == 'IMDBMulti':

            for i in range(0, self.testing_graphs.__len__() - 1):
                g2_geom = self.testing_graphs[i]
                g2_name = "test-{}".format(i)
                g2 = utils.PyG2NetworkxG(g2_geom, aids=aids)
                if g2.number_of_nodes() > graphs_size:
                    if self.args.dataset == 'IMDBMulti':
                        pass
                    else:
                        continue

                d = self.ged_matrix[i + self.training_graphs.__len__()][j]
                n = np.inf if self.args.dataset == 'IMDBMulti' else utils.calculate_ged_NX(g1, g2, aids=aids)

                if withPrediction:
                    p = utils.denormalize_sim_score(g1_geom, g2_geom, self.norm_prediction_mat[i][j])
                else:
                    p = None

                utils.print2graphsNX(g1, g2, g1_name, g2_name, showGED=True, saveToFile=True, root=path,
                                     datasetGED=d, nxGED=n, pred=p, aids=aids)

                if entireDataset is False:
                    num_of_figs = num_of_figs - 1

                if entireDataset is False:
                    if num_of_figs == 0: break

        if samplePerTrainGraph:
            num_of_figs = 6
        if entireDataset is False:
            if num_of_figs == 0: break
    # end for


def lossPlot(train_loss, val_loss, args):
    plt.plot(train_loss, label="Train")
    plt.plot([*range(0, args.epochs, 10)], val_loss, label="Validation")
    plt.ylim([0, 0.01])
    plt.legend()
    filename = "../" + args.dataset
    filename += '_' + args.gnn_operator
    if args.diffpool:
        filename += '_diffpool'
    if args.histogram:
        filename += '_hist'
    filename = filename + '_epochs' + str(args.epochs) + '.pdf'
    plt.savefig(filename)

    return


def histogram(data, dataset_name, path=None):
    data = data.flatten()

    fig = plt.figure(figsize=(15, 10))
    #bins = len(set(data))
    # First the full histogram
    fig.clf()


    # Add axes and data
    ax = fig.add_subplot()
    h = ax.hist(data, bins=500, color='r')

    # Add decorations
    img_title = "Distribution of absolute pair differences in vectorized vs loop-based SimGNN predictions ({})"\
                .format(dataset_name)
    ax.set_title(img_title, fontweight="bold")
    ax.set_ylabel("Pair count")
    ax.set_xlabel("Difference size")

    if path is not None:
        utils.saveToFile(fig, img_title, path, param_type='plot')
    else:
        fig.show()

    # Now zoomed in version
    fig.clf()

    # Add axes and data
    ax = fig.add_subplot()
    h = ax.hist(data, bins=500)

    # Add decorations
    img_title = "Distribution of absolute pair differences in v" \
                "ectorized vs loop-based SimGNN predictions ({}) (Zoom in)".format(dataset_name)
    ax.set_title(img_title, fontweight="bold")
    ax.set_ylabel("Pair count")
    ax.set_xlabel("Difference size")

    error_bins = h[1]

    # Color code zoomed in histogram like the heatmap
    # We'll color code by height, but you could use any scalar
    fracs = error_bins / error_bins.max()
    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())
    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, h[2].patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    # Determine zoom level
    arr = h[0]
    arr = np.sort(arr)
    ylim = arr[-2] + 50

    ax.set_ylim(0, ylim)

    if path is not None:
        utils.saveToFile(fig, img_title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)


def heatmap(data, dataset_name, path=None):
    # Create a figure
    fig = plt.figure(figsize=(15, 10))
    fig.clf()

    # Add axes and data
    heatmap = fig.add_subplot()
    data_mappable = heatmap.imshow(data)

    # Create colorbar and add it
    def color_Axes(axes):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        return make_axes_locatable(axes).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(data_mappable, cax=color_Axes(heatmap))

    # Remove axis from heatmap
    for edge, spine in heatmap.spines.items():
        spine.set_visible(False)

    # Add decorations
    img_title = "Absolute differences in vectorized vs loop-based SimGNN predictions ({})".format(dataset_name)
    heatmap.set_title(img_title, fontweight="bold")
    heatmap.set_ylabel("test graphs")
    heatmap.set_xlabel("train graphs")
    fig.tight_layout()

    # Show or save
    if path is not None:
        utils.saveToFile(fig, img_title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)


def print2graphsNX(g1, g2, g1_name, g2_name, showGED=False, saveToFile=False, root=None, datasetGED=None, nxGED=None,
                   pred=None, aids=False):
    '''
    Function that takes as arguments 2 networkx graphs, and their names, and prints them side by side. The figure can contain GED
    values that are precomputed, or computed here by Networkx library. The figure can be presented or saved as image file.
    :param g1: The first graph to print. It will be printed on the left and in red color.
    :param g2: The second graph to print. It will be printed on the right and in blue color.
    :param g1_name: The first graph's name to print as title.
    :param g2_name: The second graph's name to print as title.
    :param showGED: Boolean value to determine if the figure will contain the GED value. Default value is False.
    :param saveToFile: Boolean value to determine if the figure will be saved as a .png image or only presented. Default
    value is False.
    :param root: The relative path used to save the image if `saveToFile` is True.
    :param datasetGED: Variable to hold a GED value that is precalculated or existing in the dataset. If such a value
    exists assing it here, else the default value is None. `showGED` must be True in order to be used.
    :param nxGED: Same as `datasetGED`. The difference is that this variable represents GED computed through NetworkX
    library. Default value is None, otherwise the expected value is numeric. `showGED` must be True in order to be used.
    :return: Nothing
    '''
    if saveToFile:
        # matplotlib.get_backend()
        # matplotlib.use('Agg')
        pass

    g1_name = "graph " + g1_name
    g2_name = "graph " + g2_name

    fig = plt.figure(figsize=(10, 10))
    fig.clf()

    ax1 = fig.add_subplot(121, frame_on=False)
    ax2 = fig.add_subplot(122, frame_on=False)

    ax1.set_title(g1_name, fontsize=16, fontweight="bold")
    ax2.set_title(g2_name, fontsize=16, fontweight="bold")

    labels = {}
    labels1 = {}
    if aids:
        for i, _ in enumerate(g1.nodes):
            labels[i] = g1.nodes[i]['label']
        for i, _ in enumerate(g2.nodes):
            labels1[i] = g2.nodes[i]['label']

    nx.draw_networkx(g1, ax=ax1, node_size=600, width=2, font_size=14, labels=labels, node_color='r')
    nx.draw_networkx(g2, ax=ax2, node_size=600, width=2, font_size=14, labels=labels1)

    if showGED:
        n = nxGED if nxGED is not None else utils.calculate_ged_NX(g1, g2, aids)
        figure_text = "Ground Truth (Networkx) = {}"
        if datasetGED is not None:
            figure_text = figure_text + "\nGround Truth (Dataset) = {}"
        if pred is not None:
            figure_text = figure_text + "\nPrediction = {}"

        plt.figtext(x=0.4, y=0.03, s=figure_text.format(n,
                                                        datasetGED if datasetGED is not None else [],
                                                        pred if pred is not None else []))

    if saveToFile:
        root = root + "/{}_{}.png"
        plt.savefig(root.format(g1_name, g2_name))  # save as png
    else:
        fig.show()


def draw_graphs(glist, aids=False):
    for i, g in enumerate(glist):
        plt.clf()
        G = to_networkx(g).to_undirected()
        if aids:
            label_list = utils.aids_labels(g)
            labels = {}
            for j, node in enumerate(G.nodes()):
                labels[node] = label_list[j]
            nx.draw(G, labels=labels)
        else:
            nx.draw(G)
        plt.savefig('graph{}.png'.format(i))


def draw_weighted_nodes(filename, g, model):
    """
    Draw graph with weighted nodes (for AIDS). Visualizations of node attentions. The darker the
    color, the larger the attention weight.
    """
    features = model.convolutional_pass(g.edge_index, g.x)
    coefs = model.attention.get_coefs(features)

    print(coefs)

    plt.clf()
    G = to_networkx(g).to_undirected()

    label_list = utils.aids_labels(g)
    labels = {}
    for i, node in enumerate(G.nodes()):
        labels[node] = label_list[i]

    vmin = coefs.min().item() - 0.005
    vmax = coefs.max().item() + 0.005

    nx.draw(G, node_color=coefs.tolist(), cmap=plt.cm.Reds, labels=labels, vmin=vmin, vmax=vmax)

    # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm.set_array(coefs.tolist())
    # cbar = plt.colorbar(sm)

    plt.savefig(filename)


def comparativeDistribution(withLSH, noLSH, dataset_name, path, address="Global"):
    import statistics as stat

    # Create a figure
    fig = plt.figure(figsize=(15, 10))
    fig.clf()

    # Add axes and data
    ax_with = fig.add_subplot(121)
    ax_without = fig.add_subplot(122)

    h_with = ax_with.hist(withLSH, bins=100, color='#e09d4b', zorder=3)
    h_without = ax_without.hist(noLSH, bins=100, color='#58a066', zorder=3)

    # Add specified limits to the axis
    arr = h_with[0]
    arr = np.sort(arr)
    y1 = arr[-1]
    arr = h_without[0]
    arr = np.sort(arr)
    y2 = arr[-1]

    ymax = max(y1, y2) + max(y1, y2)/2
    ymin = 0
    ax_with.set_ylim([ymin, ymax])
    ax_without.set_ylim([ymin, ymax])
    try:
        xmax = max(max(withLSH), max(noLSH)) + 0.05
    except ValueError:
        print("stop)")
    xmin = -0.05
    ax_with.set_xlim([xmin, xmax])
    ax_without.set_xlim([xmin, xmax])

    # Add decorations
    img_title = "({}) Prediction errors with&without LSH ({})".format(address, dataset_name)
    fig.suptitle(img_title, fontweight="bold")
    import statistics
    try:
        ax_with.set_title('With LSH, variance={}'.format(round(stat.variance(withLSH), 5)))
        ax_without.set_title('Without LSH, variance={}'.format(round(stat.variance(noLSH), 5)))
    except statistics.StatisticsError:
        print("wow")
    ax_with.set_ylabel('Count')
    ax_with.set_xlabel('Errors')
    ax_without.set_xlabel('Errors')
    ax_with.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)
    ax_without.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    if path is not None:
        utils.saveToFile(fig, img_title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)


def comparativeScatterplot(withLSH, noLSH, dataset_name, path, address="Global"):
    import statistics as stat

    fig = plt.figure(figsize=(15, 10))
    fig.clf()

    # Add axes and data
    scat = fig.add_subplot()
    scat.scatter(x=withLSH, y=noLSH, zorder=3)

    # Add specified limits to the axis
    xmax = max(withLSH)
    xmin = min(withLSH)

    ymax = max(noLSH)
    ymin = min(noLSH)

    #scat.set_xlim([xmin, xmax])
    #scat.set_ylim([ymin, ymax])

    # Add decorations
    plt.plot(np.linspace(0, xmax, 10), np.linspace(0, xmax, 10), c="red", linestyle=':', zorder=4)

    title = "({}) Prediction errors with & without LSH ({})".format(address, dataset_name)
    scat.set_title(title, fontweight="bold")
    scat.set_ylabel('Errors WITHOUT LSH')
    scat.set_xlabel('Errors WITH LSH')
    scat.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    if path is not None:
        utils.saveToFile(fig, title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)

def drillDownSSE(labels, sse, dataset_name, path):
    fig = plt.figure(figsize=(15, 10))
    fig.clf()

    # Add axes and data
    bars = fig.add_subplot()

    x = np.arange(start=1, stop=len(labels)+1)  # the label locations
    width = 0.5  # the width of the bars
    bars.bar(x=x, height=sse, width=width, color=['#ffb59b'], label='SSE', zorder=3, edgecolor='black')

    # Add decorations
    bars.set_ylabel('Scores')
    title = "Sum of squared errors (SSE) with LSH - Globally and in Buckets ({}).".format(dataset_name)
    bars.set_title(title, fontweight="bold")
    bars.set_xticks(x)
    bars.set_xticklabels(labels, rotation=45, ha='right')
    bars.legend()
    bars.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    fig.tight_layout()

    if path is not None:
        utils.saveToFile(fig, title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)

def drillDownMSE(labels, mse, dataset_name, path):
    fig = plt.figure(figsize=(15, 10))
    fig.clf()

    # Add axes and data
    bars = fig.add_subplot()

    x = np.arange(start=1, stop=len(labels)+1)  # the label locations
    width = 0.5  # the width of the bars
    bars.bar(x=x, height=mse, width=width, color=['#8fbf7f'], label='MSE(10^-3)', zorder=3, edgecolor='black')

    # Add decorations
    bars.set_ylabel('Scores')
    title = "Mean squared errors (MSE) with LSH - Globally and in Buckets ({}).".format(dataset_name)
    bars.set_title(title, fontweight="bold")
    bars.set_xticks(x)
    bars.set_xticklabels(labels, rotation=45, ha='right')

    ylim = min(mse)
    ylim = ylim*0.8
    bars.set_ylim(bottom=ylim, top=None)

    bars.legend()
    bars.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    fig.tight_layout()

    if path is not None:
        utils.saveToFile(fig, title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)

def drillDownCorrelation(stats, dataset_name, path):
    # Note: SSE is plotted by itself in another plot
    fig = plt.figure(figsize=(15, 10))
    fig.clf()

    # Add axes and data
    bars = fig.add_subplot()

    x = np.arange(start=1, stop=len(stats["labels"])+1)  # the label locations
    width = 0.3  # the width of the bars

    #bars.bar(x=x, height=sse, width=width, color=['#ff764a'], label='SSE', zorder=3)
    #bars.bar(x - 1 * width, stats["rho"], width, color=['#dac767'], zorder=3, label="Spearman's rho")
    #bars.bar(x - 0 * width, stats["tau"], width, color=['#5f7bde'], zorder=3, label="Kendall's tau")

    bars.bar(x - 1 * width, stats["rho"], width, color=['#a2b3c8'], zorder=3, label="Spearman's rho", edgecolor='black')
    bars.bar(x - 0 * width, stats["tau"], width, color=['#eec69d'], zorder=3, label="Kendall's tau", edgecolor='black')

    # Add decorations
    bars.set_ylabel('Scores')
    title = "Correlation scores with LSH - Globally and in Buckets ({}).".format(dataset_name)
    bars.set_title(title, fontweight="bold")
    bars.set_xticks(x)
    bars.set_xticklabels(stats["labels"], rotation=45, ha='right')

    ylim = min(min(stats["rho"]), min(stats["tau"]))
    ylim = ylim*0.8
    bars.set_ylim(bottom=ylim, top=None)

    bars.legend()
    bars.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    fig.tight_layout()

    if path is not None:
        utils.saveToFile(fig, title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)

def drillDownStats2(stats, dataset_name, path):
    # Note: SSE is plotted by itself in another plot
    fig = plt.figure(figsize=(15, 10))
    fig.clf()

    # Add axes and data
    bars = fig.add_subplot()

    x = np.arange(start=1, stop=len(stats["labels"])+1)  # the label locations
    width = 0.3  # the width of the bars

    #bars.bar(x=x, height=sse, width=width, color=['#ff764a'], label='SSE', zorder=3)
    #bars.bar(x + 0 * width, stats["p10"], width, color=['#e06f45'], zorder=3, label='Precision@10')
    #bars.bar(x + 1 * width, stats["p20"], width, color=['#00876c'], zorder=3, label='Precision@20')
    #bars.bar(x + 2 * width, stats["ale"], width, color=['#58a066'], zorder=3, label='Av. Relative Error')
    bars.bar(x - 1 * width, stats["p10"], width, color=['#b8c2e8'], zorder=3, label='Precision@10', edgecolor='black')
    bars.bar(x + 0 * width, stats["p20"], width, color=['#eeb99a'], zorder=3, label='Precision@20', edgecolor='black')
    bars.bar(x + 1 * width, stats["ale"], width, color=['#a5c8a9'], zorder=3, label='Av. Relative Error', edgecolor='black')

    # Add decorations
    bars.set_ylabel('Scores')
    title = "Test scores with LSH - Globally and in Buckets ({}).".format(dataset_name)
    bars.set_title(title, fontweight="bold")
    bars.set_xticks(x)
    bars.set_xticklabels(stats["labels"], rotation=45, ha='right')
    bars.legend()
    bars.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    fig.tight_layout()

    if path is not None:
        utils.saveToFile(fig, title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)


def showLSHTablesDistributions(trainer, dataset_name, saveToFile=False):
    """
    This method plots distributional characteristics of the LSH indexing, with respect to Hash Size (K) and
    Number of Tables (L). Embedding Size set in the model parameters is also denoted.
    """
    import scipy.stats as stats

    K = [1, 3, 5, 10, 20, 30, 40, 50]
    L = [1, 2, 4, 6, 8, 10, 14, 16, 20]

    # Flag to denote if the plots have to do with change in K or L.
    HashSizeTest = False  # True False
    if HashSizeTest:
        param_in_focus = K
        plot_title = "Hash Size"
        root = "../lsh_distributions/emb{}_HashSize ({})/".format(trainer.embeddings_size, dataset_name)
    else:
        param_in_focus = L
        plot_title = "Number of LSH Tables"
        root = "../lsh_distributions/emb{}_NumOfTables ({})/".format(trainer.embeddings_size, dataset_name)
    utils.clearDataFolder(root)

    # Metrics
    mean_max_perK = []  # Mean maximum bucket size per k
    mean_dist_to_second_perK = []  # Mean distance between the 2 top buckets per k
    utilized_graphs_perK = []  # Mean number of graphs contained in buckets that are trainable

    for k in param_in_focus:
        if HashSizeTest:
            makeLSHIndex(trainer, hash_size=k)
        else:
            makeLSHIndex(trainer, num_of_tables=k)

        lsh = trainer.lsh

        # counts_total = []
        # counts_cleaned = []

        # List to hold the raw counts for the mean metrics above.
        maxes_for_hashsize_k = []
        dists_to_sec_for_hashsize_k = []
        graphs_utilized_for_hashsize_k = []

        for num, table in enumerate(lsh.hash_tables, start=1):
            img_title = "HashSize={}, Table {} of {}, EmbSize={}".format(lsh.hash_size,
                                                                         num,
                                                                         len(lsh.hash_tables),
                                                                         lsh.input_dim)
            # The number of graphs in ALL bucket of this table
            bucket_graph_counts = [len(i) for i in table.storage.values()]
            # counts_total.append(counts)

            # The number of graphs in TRAINABLE buckets of this table
            bucket_tr_graph_counts = [len(i) for i in table.storage.values() if len(i) >= lsh.min_bucket_size]
            # counts_cleaned.append(bucket_tr_graph_counts)

            # Get first and second max counts.
            max_in_table = max(bucket_graph_counts)
            try:
                second_in_table = sorted(set(bucket_graph_counts))[-2]
            except IndexError:
                # IndexError occurs when the list contains 2 elemetns. Essentialy the second largest is the min.
                second_in_table = min(bucket_graph_counts)

            maxes_for_hashsize_k.append(max_in_table)
            dists_to_sec_for_hashsize_k.append(max_in_table - second_in_table)

            # how many graphs will be used for LSH-training from this table.
            graphs_covered_in_table = np.sum(np.array(bucket_tr_graph_counts))
            graphs_utilized_for_hashsize_k.append(graphs_covered_in_table)

            # Time to plot!
            fig = plt.figure(figsize=(15, 10))
            fig.clf()

            fig.suptitle(img_title, fontweight="bold")

            ax1 = fig.add_subplot(121, frame_on=False)
            ax2 = fig.add_subplot(122, frame_on=False)

            # On the left a quantile-quantile plot on uniform distribution
            stats.probplot(bucket_graph_counts, dist="uniform", plot=ax1)

            ax1.get_lines()[0].set_linestyle('dashed')
            ax1.get_lines()[0].set_label('Complete')

            ax1.get_lines()[1].set_label('Uniform')

            # Add the 'trainable-bucket-line' if the list isn't empty
            if len(bucket_tr_graph_counts) != 0:
                stats.probplot(bucket_tr_graph_counts, dist="uniform", plot=ax1)

                ax1.get_lines()[2].set_linestyle('dashed')
                ax1.get_lines()[2].set_color('green')
                ax1.get_lines()[2].set_label('Cleaned')

            ax1.get_children()[10].set_fontweight('bold')
            ax1.legend()

            # On the right the histogram to show bucket distribution
            x = np.arange(1, len(bucket_graph_counts) + 1)
            ax2.bar(x, bucket_graph_counts, zorder=3)
            ax2.axhline(y=trainer.lsh.min_bucket_size, color='red', zorder=1, label='Trainable bucket min. size')

            ax2.set_title("Bucket Distribution", fontweight="bold")
            ax2.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)
            ax2.legend()

            if len(bucket_graph_counts) < 20:
                ax2.set_xticks(x)

            ax2.set_ylabel('Graph Number')
            ax2.set_xlabel('Buckets')

            path = root
            if path is not None:
                utils.saveToFile(fig, img_title, path, param_type='plot')
            else:
                fig.show()
            plt.close(fig)
        pass

        # Now, having finished with the tables, extract the means for the stats of this K
        mean_max_for_k = np.mean(np.array(maxes_for_hashsize_k))
        mean_max_perK.append(mean_max_for_k)

        mean_dist_to_second_for_k = np.mean(np.array(dists_to_sec_for_hashsize_k))
        mean_dist_to_second_perK.append(mean_dist_to_second_for_k)

        mean_graph_cover_for_k = np.mean(np.array(graphs_utilized_for_hashsize_k))
        utilized_graphs_perK.append(mean_graph_cover_for_k)

    # Now we plot the mean statistics for all K to check for any trend.
    figTrends = plt.figure(figsize=(15, 10))
    figTrends.clf()

    img_title = "3 LSH Tables Means over {}, EmbSize={}".format(plot_title, trainer.embeddings_size)

    plt.title(img_title, fontweight="bold")
    plt.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    plt.plot(param_in_focus, mean_max_perK, color='blue', label='Maximum bucket size (mean)')
    plt.plot(param_in_focus, mean_dist_to_second_perK, color='red', label='Distance to 2nd largest bucket (mean)')
    plt.plot(param_in_focus, utilized_graphs_perK, color='green',
             label='Number of training graphs in trainable buckets (mean)')

    plt.xlabel(plot_title, fontsize=14)
    plt.ylabel('Mean Values', fontsize=14)
    plt.xticks(param_in_focus)
    plt.legend()

    path = root
    if path is not None:
        utils.saveToFile(figTrends, img_title, path, param_type='plot')
    else:
        figTrends.show()
    plt.close(figTrends)


def LSHGEDdistribution(geds, dataset_name, path=None):
    #data = data.flatten()

    fig = plt.figure(figsize=(15, 10))
    # bins = len(set(data))
    # First the full histogram
    fig.clf()

    maxged = max(geds)+1
    # Add axes and data
    ax = fig.add_subplot()
    h = ax.hist(geds, bins=np.arange(maxged+1) - 0.5, color='#89b9a9', edgecolor='black', hatch='/', zorder=3)

    # Add decorations
    img_title = "Distribution of LSH utilization based on GED ({})" \
        .format(dataset_name)
    ax.set_title(img_title, fontweight="bold")
    ax.set_ylabel("Pair count")
    ax.set_xlabel("GED values")
    ax.set_xticks(range(maxged))
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, zorder=0)

    if path is not None:
        utils.saveToFile(fig, img_title, path, param_type='plot')
    else:
        fig.show()
    plt.close(fig)
    return None