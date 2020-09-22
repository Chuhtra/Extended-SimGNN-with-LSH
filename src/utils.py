import torch
import numpy as np
import math
from texttable import Texttable
import networkx as nx
from torch_geometric.utils import to_networkx

##-----------------------------------------------------files
def fixpath(path):
    # Create path directories if missing
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

def saveToFile(data, nameOfFile, path, param_type=None):
    fixpath(path)

    filename = path + nameOfFile
    if param_type is None:  # default is nd array
        # open a binary file in write mode
        file = open(filename, "wb")
        # save array to the file
        np.save(file, data)
    elif param_type == 'dict' or param_type == 'str':  # save param_type to pickle
        import pickle
        filename = filename + '.pkl'
        # open a binary file in write mode
        file = open(filename, "wb")
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    elif param_type == 'csv':
        import csv
        file = open(filename + '.csv', "w", newline='')
        writer = csv.writer(file)
        writer.writerows(data)
    elif param_type == 'plot':
        fixpath(path)
        path += nameOfFile + ".png"
        data.savefig(path, bbox_inches='tight')  # save as png
        return
    # close the file
    file.close()


def readFromFile(nameOfFile, path, param_type=None):
    filename = path + nameOfFile
    if param_type is None:  # default is nd array
        # open the file in read binary mode
        file = open(filename, "rb")
        # read the file to numpy array
        data = np.load(file, allow_pickle=True)
    elif param_type == 'dict' or param_type == 'str':  # load pickle to param_type
        import pickle
        filename = filename + '.pkl'
        # open the file in read binary mode
        file = open(filename, "rb")
        data = pickle.load(file)
    # close the file
    file.close()

    return data

def clearDataFolder(path):
    import os, shutil
    try:
        folder = os.listdir(path)
    except FileNotFoundError:
        return
    else:
        if len(folder) != 0:
            for filename in folder:
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

##-----------------------------------------------------printer
def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model - Namespace or dictionary.
    """
    # This if-else distiction is done for LSH variables as a temporary solution. Because they are not printed at the
    # start.
    if isinstance(args, dict):
        title = 'LSH '
    else:
        title = ''
        args = vars(args)

    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["{}Parameter".format(title), "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])

    print("\n")
    print(t.draw())
    print("\n")

##-----------------------------------------------------math
def getSSE(dif):
    return np.sum(np.square(dif))

def getAvRelEr(dif, target):
    return np.mean(np.abs(dif) / target)


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))

    return rank_corr_function(r_prediction, r_target).correlation


def calculate_prec_at_k(k, prediction, target):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[:k]
    best_k_target = target.argsort()[:k]

    return len(set(best_k_pred).intersection(set(best_k_target))) / k

def denormalize_sim_score(g1, g2, sim_score):
    """
    Converts normalized similarity into ged.
    """
    return denormalize_ged(g1, g2, -math.log(sim_score, math.e))


def denormalize_ged(g1, g2, nged):
    """
    Converts normalized ged into ged.
    """
    return round(nged * (g1.num_nodes + g2.num_nodes) / 2) if nged != np.inf else np.inf


##-----------------------------------------------------graph
def PyG2NetworkxG(g, aids=False):
    y = to_networkx(g, to_undirected=(True if g.is_undirected() else False))
    if aids:
        label_list = aids_labels(g)
        for i, _ in enumerate(y.nodes):
            y.nodes[i]['label'] = label_list[i]
    # for i in y.nodes:
    #    y.nodes[i]['name'] = i
    return y

def aids_labels(g):
    types = [
        'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
        'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
        'Sb', 'Se', 'Ni', 'Te'
    ]

    return [types[i] for i in g.x.argmax(dim=1).tolist()]

# This is the function which checks for equality of labels
def node_match_equality(node1, node2):
    x = node1['label'] == node2['label']
    return x

def calculate_ged_NX(g1, g2, aids=False):
    if aids:
        x = node_match_equality
    else:
        x = None
    return nx.graph_edit_distance(g1, g2, node_match=x)

def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)
