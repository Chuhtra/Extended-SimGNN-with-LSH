import argparse
import torch

def initGlobals(dataset_name):
    """
    Used to initialize some global variables that help with saving result files.
    """
    global temp_runfiles_path
    global global_res_path
    global lsh_res_path
    global vector_loopy_res_path

    temp_runfiles_path = '../temp_runfiles ({})/'.format(dataset_name)

    global_res_path = temp_runfiles_path + "global_results/"
    lsh_res_path = temp_runfiles_path + "lsh_results/"
    vector_loopy_res_path = temp_runfiles_path + "vectorizedVSloopy/"



def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument("--dataset",
                        nargs="?",
                        default="AIDS700nef",  # AIDS700nef LINUX IMDBMulti
                        help="Dataset name. Default is AIDS700nef")

    parser.add_argument("--gnn-operator",
                        nargs="?",
                        default="gin",  # gcn gin gat
                        help="Type of GNN-Operator. Default is gcn")

    parser.add_argument("--epochs",
                        type=int,
                        default=350,
                        help="Number of training epochs. Default is 350.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 1st convolution. Default is 64.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=32,
                        help="Filters (neurons) in 2nd convolution. Default is 32.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=32,  ##
                        help="Filters (neurons) in 3rd convolution. Default is 32.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
                        help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
                        help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
                        help="Histogram Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0,
                        help="Dropout probability. Default is 0.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5 * 10 ** -4,
                        help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.add_argument("--diffpool",
                        dest="diffpool",
                        action="store_true",
                        help="Enable differentiable pooling.")

    parser.add_argument("--plot-loss",
                        dest="plot_loss",
                        action="store_true")

    parser.add_argument("--notify",
                        dest="notify",
                        action="store_true",
                        help="Send notification message when the code is finished (only Linux & Mac OS support).")

    # TODO device selection
    #parser.add_argument("--device",
    #                    nargs="?",
    #                    default='cpu',  # torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    #                    help="Select to run with gpu or cpu. Default depends on existing CUDA installation.")

    parser.add_argument("--use-lsh",
                        dest="use_lsh",
                        action="store_true",
                        help="Specify if LSH will be utilized. Default choice is to train WITH LSH.")


    parser.set_defaults(histogram=False)  # True False
    parser.set_defaults(use_lsh=False)  # True False
    parser.set_defaults(diffpool=False)  # True False
    parser.set_defaults(plot_loss=False)  # True False
    parser.set_defaults(notify=False)  

    # TODO add lsh variables as arguments conditional on --use-lsh

    return parser.parse_args()
