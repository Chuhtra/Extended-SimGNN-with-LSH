import torch
import numpy as np
import utils
import tests

import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from utils import calculate_ranking_correlation, calculate_prec_at_k
import plottingFunctions as plot
import param_parser as param


#import sys
#sys.stdout = open("../file.txt", "w+")

def getGlobalGEDDifs(norm_ged_index):
    estimated_ged_lsh = []
    estimated_ged_nolsh = []  # np.array((len(self.testing_graphs) * len(self.training_graphs)))
    true_ged = []

    for i in norm_ged_index:
        for j in i:
            if j["lsh_use"]:
                true_ged.append(j["target"])
                estimated_ged_lsh.append(j["post_prediction"])
                estimated_ged_nolsh.append(j["prior_prediction"])

    true_ged = np.array(true_ged)

    estimated_ged_lsh = np.array(estimated_ged_lsh)
    ged_dif_lsh = estimated_ged_lsh - true_ged

    estimated_ged_nolsh = np.array(estimated_ged_nolsh)
    ged_dif_nolsh = estimated_ged_nolsh - true_ged

    return ged_dif_lsh, ged_dif_nolsh, true_ged


def print_evaluation(metrics, errors=None):
    """
    Printing the error rates.
    """
    print("\nmse(10^-3): " + str(round(metrics[0] * 1000, 5)) + ".")
    print("Spearman's rho: " + str(round(metrics[1], 5)) + ".")
    print("Kendall's tau: " + str(round(metrics[2], 5)) + ".")
    print("p@10: " + str(round(metrics[3], 5)) + ".")
    print("p@20: " + str(round(metrics[4], 5)) + ".")

    if errors is not None:
        for i in errors:
            print(i)


def printPriors():
    # PRINT prior VECTORIZED
    vectorized_priors = [
        utils.readFromFile("global_mse", param.global_res_path).item(),
        utils.readFromFile("global_rho", param.global_res_path).item(),
        utils.readFromFile("global_tau", param.global_res_path).item(),
        utils.readFromFile("global_p@10", param.global_res_path).item(),
        utils.readFromFile("global_p@20", param.global_res_path).item()
    ]
    print("\nWithout LSH (VECTORIZED):")
    print_evaluation(vectorized_priors)


def vectorizedVSloopy():
    for i in utils.readFromFile("zeroDifsCounts", param.vector_loopy_res_path):
        print(i)

    differences = utils.readFromFile("AbsolutePiorDifs", param.vector_loopy_res_path)

    printPriors()

    # Prior loopy is printed in the next section

    plot.heatmap(differences, dataset_name, path=param.vector_loopy_res_path + plot_subfolder_name)
    plot.histogram(differences, dataset_name, path=param.vector_loopy_res_path + plot_subfolder_name)


def priorVSposterior():
    norm_ged_index = utils.readFromFile("global_norm_ged_lsh_index", param.global_res_path)
    ged_dif_lsh, ged_dif_nolsh, true_ged = getGlobalGEDDifs(norm_ged_index)

    # PRINT prior LOOPY
    loopy_priors = [
        utils.readFromFile("global_prior_mse", param.global_res_path).item(),
        utils.readFromFile("global_prior_rho", param.global_res_path).item(),
        utils.readFromFile("global_prior_tau", param.global_res_path).item(),
        utils.readFromFile("global_prior_p@10", param.global_res_path).item(),
        utils.readFromFile("global_prior_p@20", param.global_res_path).item()
    ]
    SSE_noLSH = utils.getSSE(ged_dif_nolsh)  # Sum of squared errors
    AVG_REL_ERROR_noLSH = utils.getAvRelEr(ged_dif_nolsh, true_ged)  # Average relative error
    prior_errors = ["\nSSE (no LSH) = {}".format(SSE_noLSH),
                    "AVG_REL_ERROR (no LSH) = {}".format(AVG_REL_ERROR_noLSH)]
    print("\nWithout LSH (Loop-based):")
    print_evaluation(metrics=loopy_priors, errors=prior_errors)

    # #PRINT POSTERIOR
    loopy_posts = [
        utils.readFromFile("global_post_mse", param.global_res_path).item(),
        utils.readFromFile("global_post_rho", param.global_res_path).item(),
        utils.readFromFile("global_post_tau", param.global_res_path).item(),
        utils.readFromFile("global_post_p@10", param.global_res_path).item(),
        utils.readFromFile("global_post_p@20", param.global_res_path).item()
    ]
    SSE_LSH = utils.getSSE(ged_dif_lsh)  # Sum of squared errors
    AVG_REL_ERROR_LSH = utils.getAvRelEr(ged_dif_lsh, true_ged)  # Average relative error
    post_errors = ["\nSSE (LSH) = {}".format(SSE_LSH),
                   "AVG_REL_ERROR (LSH) = {}".format(AVG_REL_ERROR_LSH)]
    print("\nWith LSH:")
    print_evaluation(metrics=loopy_posts, errors=post_errors)

    # Now, Global distribution and variance of errors.
    plot.comparativeDistribution(np.abs(ged_dif_lsh), np.abs(ged_dif_nolsh), dataset_name,
                                 path=param.global_res_path + plot_subfolder_name)
    plot.comparativeScatterplot(np.abs(ged_dif_lsh), np.abs(ged_dif_nolsh), dataset_name,
                                path=param.global_res_path + plot_subfolder_name)

    return loopy_posts, SSE_LSH, AVG_REL_ERROR_LSH


def lshUtilizationCounts():
    for i in (utils.readFromFile("utilization_counts", param.lsh_res_path)):
        print(i)

    norm_ged_index = utils.readFromFile("global_norm_ged_lsh_index", param.global_res_path)

    pair_geds = []
    for i in norm_ged_index:
        for j in i:
            if j["lsh_use"]:
                pair_geds.append(j["target_denorm"])

    plot.LSHGEDdistribution(pair_geds, dataset_name, path=param.lsh_res_path)


def drillDownBuckets(drillDownStats):
    path_for_dd_plots = param.global_res_path + 'drill_down_' + plot_subfolder_name
    trainable_buckets = utils.readFromFile("trainable_buckets_dict", param.lsh_res_path,
                                           param_type='dict')
    dd_index = utils.readFromFile("drill_down_index", param.lsh_res_path)

    bucketpriors = [[] for _ in range((len(trainable_buckets["bucketName"])))]
    bucketposts = [[] for _ in range((len(trainable_buckets["bucketName"])))]
    buckettargets = [[] for _ in range((len(trainable_buckets["bucketName"])))]

    for index_j, drill_dict in enumerate(dd_index):
        bucketpriors[drill_dict["bucket_index"]].append(drill_dict["priorpred"])
        bucketposts[drill_dict["bucket_index"]].append(drill_dict["postpred"])
        buckettargets[drill_dict["bucket_index"]].append(drill_dict["target"])


    for i, b in enumerate(trainable_buckets["bucketName"]):
        prior = np.array(bucketpriors[i])
        ground = np.array(buckettargets[i])
        post = np.array(bucketposts[i])

        # if there are unutilized buckets they should be skipped
        if len(prior) == 0: continue

        # later stats
        drill_ged_dif_lsh = post - ground
        drill_ged_dif_nolsh = prior - ground

        # Prior paper stats
        prior_drills = [
            np.mean(F.mse_loss(torch.tensor(prior), torch.tensor(ground), reduction='none').detach().numpy()),
            calculate_ranking_correlation(spearmanr, prior, ground),
            calculate_ranking_correlation(kendalltau, prior, ground),
            calculate_prec_at_k(10, prior, ground),
            calculate_prec_at_k(20, prior, ground)
        ]
        SSE_noLSH_drill = utils.getSSE(drill_ged_dif_nolsh)
        AVG_REL_ERROR_noLSH_drill = utils.getAvRelEr(drill_ged_dif_nolsh, ground)
        prior_errors_drill = ["\nSSE (no LSH) = {}".format(SSE_noLSH_drill),
                              "AVG_REL_ERROR (no LSH) = {}".format(AVG_REL_ERROR_noLSH_drill)]
        print("\nTable {}, bucket {} ({})".format(trainable_buckets["table"][i], b, int(b, 2)))
        print("\nWITHOUT LSH:")
        print_evaluation(prior_drills, prior_errors_drill)

        # Post paper stats
        scoresDRILLPOST = np.mean(F.mse_loss(torch.tensor(post), torch.tensor(ground), reduction='none')
                                  .detach().numpy())
        rho_listDRILLPOST = calculate_ranking_correlation(spearmanr, post, ground)
        tau_listDRILLPOST = calculate_ranking_correlation(kendalltau, post, ground)
        prec_at_10_listDRILLPOST = calculate_prec_at_k(10, post, ground)
        prec_at_20_listDRILLPOST = calculate_prec_at_k(20, post, ground)

        SSE_LSH_drill = utils.getSSE(drill_ged_dif_lsh)
        AVG_REL_ERROR_LSH_drill = utils.getAvRelEr(drill_ged_dif_lsh, ground)
        post_errors_drill = ["\nSSE (LSH) = {}".format(SSE_LSH_drill),
                             "AVG_REL_ERROR (LSH) = {}".format(AVG_REL_ERROR_LSH_drill)]
        print("\nWITH LSH:")
        print_evaluation([scoresDRILLPOST, rho_listDRILLPOST, tau_listDRILLPOST,
                          prec_at_10_listDRILLPOST, prec_at_20_listDRILLPOST],
                         post_errors_drill)

        # For bar chart
        label = "Table {}, bucket {}".format(trainable_buckets["table"][i], int(b, 2))

        drillDownStats["labels"].append(label)
        drillDownStats["mse"].append(scoresDRILLPOST)
        drillDownStats["rho"].append(rho_listDRILLPOST)
        drillDownStats["tau"].append(tau_listDRILLPOST)
        drillDownStats["p10"].append(prec_at_10_listDRILLPOST)
        drillDownStats["p20"].append(prec_at_20_listDRILLPOST)
        drillDownStats["sse"].append(SSE_LSH_drill)
        drillDownStats["ale"].append(AVG_REL_ERROR_LSH_drill)

        # Error distribution

        plot.comparativeDistribution(np.abs(drill_ged_dif_lsh), np.abs(drill_ged_dif_nolsh), dataset_name,
                                     path=path_for_dd_plots, address=label)
        plot.comparativeScatterplot(np.abs(drill_ged_dif_lsh), np.abs(drill_ged_dif_nolsh), dataset_name,
                                    path=path_for_dd_plots, address=label)

        # LSH Utilization
        # used_pairs = len(prior)
        # how will I get bucket size?
        # bucket_pairs = 0
        # pairspercent = round(used_pairs * 100 / bucket_pairs, 1)
        # print("\nLSH Usage (pairs): {} of {} ({}%)".format(used_pairs, bucket_pairs, pairspercent))

    # Now we plot the drill down bar chart WITH LSH
    # First the SSE on its own since it's way bigger than the others.
    plot.drillDownSSE(drillDownStats["labels"], drillDownStats["sse"], dataset_name, path=path_for_dd_plots)
    plot.drillDownMSE(drillDownStats["labels"], drillDownStats["mse"], dataset_name, path=path_for_dd_plots)

    plot.drillDownCorrelation(drillDownStats, dataset_name, path=path_for_dd_plots)
    plot.drillDownStats2(drillDownStats, dataset_name, path=path_for_dd_plots)


def getDatasetName():
    d = ['AIDS700nef', 'LINUX', 'IMDBMulti']

    #num = input("Which dataset statistics? Press the number"
    #            "\n0. {} \n1. {} \n2. {}".format(d[0], d[1], d[2]))
    #dataset_name = d[int(num)]
    dataset_name = d[0]

    param.initGlobals(dataset_name)

    tests.testName(dataset_name,
                   utils.readFromFile("datasetName", param.temp_runfiles_path, param_type='str'))

    return dataset_name


def mainResults(dataset=None, withLSH=True):
    # INITIALIZATIONS
    global plot_subfolder_name
    plot_subfolder_name = 'plots/'

    global dataset_name
    dataset_name = getDatasetName() if dataset is None else dataset

    if withLSH:

        print("\nDataset used: {}".format(dataset_name))
        # 1ST: COMPARING VECTORIZED AND LOOP-BASED PRIORS
        print("#####################################################################################")
        print("########## Comparing Vectorzied & Loop-based Predictions without LSH ################")
        print("#####################################################################################")

        vectorizedVSloopy()

        # 2ND: COMPARING PRIOR AND POSTERIOR RESULTS
        # Note: Paper metrics are calculated on the entire dataset, while the rest only on the utilized pairs.

        print("#####################################################################################")
        print("########## Comparing Predictions with and without LSH - Globally ###################")
        print("#####################################################################################")
        loopy_posts, SSE_LSH_loopy, AVG_REL_ERROR_LSH_loopy = priorVSposterior()

        # 3RD: LSH UTILIZATION COUNTS
        # Pairs in LSH

        print("#####################################################################################")
        print("################### LSH Utilization Statistics - Globally ###########################")
        print("#####################################################################################")
        lshUtilizationCounts()

        # 4TH: DRILL DOWN IN BUCKETS
        # Now the drill down part - with bar chart lists
        print("#####################################################################################")
        print("########### Comparing Predictions with and without LSH - In Buckets #################")
        print("#####################################################################################")

        # barchart data - collect the final scores inside each bucket; the first elements are the globals
        drillDownStats = {
            'labels': ['Global'],
            'mse': [loopy_posts[0]],
            'rho': [loopy_posts[1]],
            'tau': [loopy_posts[2]],
            'p10': [loopy_posts[3]],
            'p20': [loopy_posts[4]],
            'sse': [SSE_LSH_loopy],
            'ale': [AVG_REL_ERROR_LSH_loopy]
        }

        drillDownBuckets(drillDownStats)

    else:

        print("#####################################################################################")
        print("##################### Just print pipeline results without LSH #######################")
        print("#####################################################################################")

        printPriors()

    print("##################################### The End #######################################")


if __name__ == "__main__":
    mainResults()
