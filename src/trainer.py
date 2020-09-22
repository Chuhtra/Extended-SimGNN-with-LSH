import torch
import networkx
import numpy as np
import torch.nn.functional as F
import tests
import plottingFunctions as plot

from tqdm import tqdm, trange
from scipy.stats import spearmanr, kendalltau

from layers import AttentionModule, TensorNetworkModule, DiffPool
import utils
from utils import calculate_ranking_correlation, calculate_prec_at_k, denormalize_sim_score

from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.data import DataLoader, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.datasets import GEDDataset
from torch_geometric.transforms import OneHotDegree

from simgnn import SimGNN
import param_parser as param

class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.embeddings_size = args.filters_3
        self.process_dataset()
        self.model, self.optimizer = self.setup_model()

    def setup_model(self, parameters=None):
        """
        Creating a SimGNN.
        """
        model = SimGNN(self.args, self.number_of_node_labels, self.number_of_edge_labels)

        if parameters is None:
            parameters = model.parameters()

        optimizer = torch.optim.Adam(parameters, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        return model, optimizer

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        '''
        # TODO test ALKANE
        # ALKANE train/test pairs are currently lacking GED precalculated values, so for compatibility I ignore the test
        # graphs provided.
        if self.args.dataset == "ALKANE":
            dataset = GEDDataset('../datasets/{}'.format(self.args.dataset), self.args.dataset, train=True)

            self.training_graphs = dataset[:90]
            self.testing_graphs = dataset[90:]

        else:
        '''
        self.training_graphs = GEDDataset('../datasets/{}'.format(self.args.dataset), self.args.dataset,
                                          train=True)  # [:560]
        self.testing_graphs = GEDDataset('../datasets/{}'.format(self.args.dataset), self.args.dataset,
                                         train=False)  # [:140]

        self.ged_matrix = self.training_graphs.ged
        self.nged_matrix = self.training_graphs.norm_ged

        # and in ndarray format for debugging
        self.ge_m = self.ged_matrix.numpy()
        self.nge_m = self.nged_matrix.numpy()

        # tests.testGEDcalclulation(self)
        # tests.testNetworkX(self)

        if self.training_graphs[0].x is None:
            max_degree = 0
            for g in self.training_graphs + self.testing_graphs:
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree

        self.number_of_node_labels = self.training_graphs.num_features
        self.number_of_edge_labels = self.training_graphs.num_edge_features

    def create_train_batches(self, train_set):
        """
        Creating suffled batches from the training graph list.
        :return batches: Zipped loaders as list.
        """

        source_loader = DataLoader(dataset=train_set.shuffle(), batch_size=self.args.batch_size)
        target_loader = DataLoader(dataset=train_set.shuffle(), batch_size=self.args.batch_size)

        return list(zip(source_loader, target_loader))

    def transform(self, data):
        """
        Getting ground truth GED for graph pairs and grouping as data into dictionary.
        :param data: Graph pair - tuple list for 1 target graph with any number of source graphs.
        :return new_data: Dictionary with data. Data contain the source graphs as g1, the target graph as g2 and the
        normalized ged (exponentiated) for each pair as target.
        """
        new_data = dict()

        new_data["g1"] = data[0]
        new_data["g2"] = data[1]

        # for each g1 and g1, access the lists with the graphs' index 'i', and retrieve the respective ged value.
        normalized_ged = self.nged_matrix[data[0]["i"].reshape(-1).tolist(),
                                          data[1]["i"].reshape(-1).tolist()].tolist()
        new_data["target"] = torch.from_numpy(np.exp([(-el) for el in normalized_ged])).view(-1).float()

        return new_data

    def process_batch(self, data, modelToUse=None):
        """
        Performs the forward pass with a batch of data.
        :param data: Data that is essentially pairs of batches, for source and target graphs.
        :return loss: Loss on the data.
        """
        model = modelToUse[0]
        optimizer = modelToUse[1]

        optimizer.zero_grad()
        data = self.transform(data)
        prediction = model(data)
        loss = F.mse_loss(prediction, data["target"], reduction='sum')
        loss.backward()
        optimizer.step()
        return loss.item()

    def fit(self, lsh_bucket=None, modelToUse=None):
        """
        Training a model.
        """
        # Unless specified, the global model is assigned.
        if modelToUse is None:
            modelToUse = (self.model, self.optimizer)

        modelToUse[0].train()

        # If no dataset is passed, then use the default training set.
        if lsh_bucket is None:
            train_set = self.training_graphs
            e = self.args.epochs
            epochs = trange(e, leave=None, desc="Epoch", position=0)
        else:
            train_set = lsh_bucket
            e = self.args.epochs  # // 2
            epochs = range(e)

        loss_list = []
        loss_list_test = []

        for epoch in epochs:
            if self.args.plot_loss and epoch % 10 == 0:  # Small validation for plotting
                loss_list_test.append(self.smallValidation(modelToUse[0]))
                modelToUse[0].train()

            batches = self.create_train_batches(train_set)
            main_index = 0
            loss_sum = 0
            # TODO: fix the printing
            # Batches progress bar is deactivated until it is fixed. Relevant to "nested bars issue"
            #_batches = tqdm(enumerate(batches), total=len(batches), desc="Batches", position=1)
            for index, batch_pair in enumerate(batches):
                loss_score = self.process_batch(batch_pair, modelToUse)
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss_score
                #_batches.update()

            loss = loss_sum / main_index
            loss_list.append(loss)

            if lsh_bucket is None:
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

        if self.args.plot_loss: plot.lossPlot(loss_list, loss_list_test, self.args)

    def smallValidation(self, model):
        """
        Perform a small validation round with some random test graphs as validation set.
        :param model:
        :return:
        """
        model.train(False)
        cnt_test = 20
        cnt_train = 100

        scores = torch.empty((cnt_test, cnt_train))

        t = tqdm(total=cnt_test * cnt_train, position=2, leave=False, desc="Validation")
        for i, g in enumerate(self.testing_graphs[:cnt_test].shuffle()):
            source_batch = Batch.from_data_list([g] * cnt_train)
            target_batch = Batch.from_data_list(self.training_graphs[:cnt_train].shuffle())
            data = self.transform((source_batch, target_batch))
            target = data["target"]
            prediction = model(data)

            scores[i] = F.mse_loss(prediction, target, reduction='none').detach()
            t.update(cnt_train)

        t.close()
        return scores.mean().item()

    def globalScore(self):
        """
        Simple scoring, without LSH.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()

        # tests.debugModelStateDict(self.model)

        # filled with exponentiated normalized ged values
        self.norm_ground_truth = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        self.norm_prediction_mat = np.empty((len(self.testing_graphs), len(self.training_graphs)))

        self.scores = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        self.rho_list = []
        self.tau_list = []
        self.prec_at_10_list = []
        self.prec_at_20_list = []

        t = tqdm(total=len(self.testing_graphs) * len(self.training_graphs))

        for i, g in enumerate(self.testing_graphs):
            # source batch is a batch with 1 test graph repeated multiple times
            source_batch = Batch.from_data_list([g] * len(self.training_graphs))
            # target batch is a batch with every training graph
            target_batch = Batch.from_data_list(self.training_graphs)

            # Get ground truth
            data = self.transform((source_batch, target_batch))
            target = data["target"]
            self.norm_ground_truth[i] = target

            # Get prediction
            prediction = self.model(data)
            self.norm_prediction_mat[i] = prediction.detach().numpy()

            # Update metrics
            self.scores[i] = F.mse_loss(prediction, target, reduction='none').detach().numpy()
            self.rho_list.append(
                calculate_ranking_correlation(spearmanr, self.norm_prediction_mat[i], self.norm_ground_truth[i]))
            self.tau_list.append(
                calculate_ranking_correlation(kendalltau, self.norm_prediction_mat[i], self.norm_ground_truth[i]))
            self.prec_at_10_list.append(calculate_prec_at_k(10, self.norm_prediction_mat[i], self.norm_ground_truth[i]))
            self.prec_at_20_list.append(calculate_prec_at_k(20, self.norm_prediction_mat[i], self.norm_ground_truth[i]))

            t.update(len(self.training_graphs))

        # Calculate metrics
        self.model_error = np.mean(self.scores)
        self.rho = np.mean(self.rho_list)
        self.tau = np.mean(self.tau_list)
        self.prec_at_10 = np.mean(self.prec_at_10_list)
        self.prec_at_20 = np.mean(self.prec_at_20_list)

        # paperMetrics = [self.model_error, self.rho, self.tau, self.prec_at_10, self.prec_at_20]
        # self.print_evaluation(paperMetrics)
        # self.presentGraphsWithGEDs(entireDataset=False, withPrediction=True)

        utils.saveToFile(self.norm_ground_truth, "global_norm_target", param.temp_runfiles_path+"global_results/")
        utils.saveToFile(self.norm_prediction_mat, "global_norm_prediction", param.temp_runfiles_path+"global_results/")
        utils.saveToFile(self.model_error, "global_mse", param.temp_runfiles_path+"global_results/")
        utils.saveToFile(self.rho, "global_rho", param.temp_runfiles_path+"global_results/")
        utils.saveToFile(self.tau, "global_tau", param.temp_runfiles_path+"global_results/")
        utils.saveToFile(self.prec_at_10, "global_p@10", param.temp_runfiles_path+"global_results/")
        utils.saveToFile(self.prec_at_20, "global_p@20", param.temp_runfiles_path+"global_results/")

    def lshScore(self):
        """
        Scoring, with LSH. Not vecotrized, since each pair my be tested by a different model.
        """
        print("\nNew scoring is initiated.")

        # INITIALIZATIONS
        csv_list = [["Graph 1 (test)", "Graph 2 (train)", "GED (True)", "GED (w/out LSH)", "GED (with LSH)"]]

        # Each cell holds a dictionary with ground truth, simgnn prediction and lsh enabled prediction. LSH use is a
        # flag. This array is used for final global results.
        norm_ged_index = [[{"target": None,
                             "prior_prediction": None,
                             "post_prediction": [],
                            "target_denorm" : None,
                             "lsh_use": None
                             } for _ in range(len(self.training_graphs))
                            ] for _ in range(len(self.testing_graphs))]

        # Each element is a dictionary that holds each prediction/target/priorpred along with the respective bucket it
        # came from.
        drill_down_index = []

        scoresPRIOR = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        rho_listPRIOR = []
        tau_listPRIOR = []
        prec_at_10_listPRIOR = []
        prec_at_20_listPRIOR = []


        scoresPOST = np.empty((len(self.testing_graphs), len(self.training_graphs)))
        rho_listPOST = []
        tau_listPOST = []
        prec_at_10_listPOST = []
        prec_at_20_listPOST = []

        pairs_not_used = 0  # In LSH
        pairsUtilizingManyTables = 0

        # These are for debuggind too
        self.predDifsNoLSH = np.zeros((len(self.testing_graphs), len(self.training_graphs)))
        self.targetDifsNoLSH = np.zeros((len(self.testing_graphs), len(self.training_graphs)))
        zeroPredDifs = 0
        zeroTargDifs = 0

        # self.debugModelsIDs()
        #

        tq = tqdm(total=len(self.testing_graphs) * len(self.training_graphs))
        # For each testing graph
        for i, testgraph in enumerate(self.testing_graphs):
            # Extract its embedding
            test_emb, _, _ = self.model.get_embedding(testgraph)
            test_emb = test_emb.detach().numpy().reshape(self.embeddings_size)

            # self.debugVectorizedTesting(testgraph)

            for j, traingraph in enumerate(self.training_graphs):

                train_emb, _, _ = self.model.get_embedding(traingraph)
                train_emb = train_emb.detach().numpy().reshape(self.embeddings_size)

                # Get ground truth
                data = self.transform((testgraph, traingraph))


                norm_ged_index[i][j]["target"] = data["target"].item()
                norm_ged_index[i][j]["target_denorm"] = utils.denormalize_sim_score(testgraph,
                                                                                    traingraph, data["target"].item())

                ## These are for debugging
                self.targetDifsNoLSH[i][j] = data["target"].item() - self.norm_ground_truth[i][j]
                if self.targetDifsNoLSH[i][j] == 0:
                    zeroTargDifs += 1

                # self.debugModelStateDict(self.model)
                ##

                # Get global prediction (no LSH)
                self.model.eval()
                prior_pred = self.model(data)
                norm_ged_index[i][j]["prior_prediction"] = prior_pred.item()

                ## These are for debugging
                self.predDifsNoLSH[i][j] = prior_pred.item() - self.norm_prediction_mat[i][j]
                if self.predDifsNoLSH[i][j] == 0:
                    zeroPredDifs += 1
                ##

                ### Now LSH testing ###
                for t, curr_table in enumerate(self.lsh.hash_tables):  # for one table at a time

                    # Hold the indexes of the trainable buckets from this table (from 'trainable buckets' dictionary)
                    trainable_bucket_indexes = [num1 for num1, i in enumerate(self.trainable_buckets["table"]) if
                                                i == t]

                    # Extract this table's test graph hashing
                    testGraphHash = self.lsh._hash(self.lsh.uniform_planes[t], test_emb)

                    # Check if this test graph is indexed in any trainable bucket. If yes, which.
                    bucket_of_interest = None
                    for b in trainable_bucket_indexes:
                        bucketHash = self.trainable_buckets["bucketName"][b]
                        if bucketHash == testGraphHash:
                            bucket_of_interest = b
                            break

                    if bucket_of_interest is not None:  # If a bucket is found:
                        # Check if the train graph falls in too, and give new prediction
                        if self.lsh._hash(self.lsh.uniform_planes[t], train_emb) == testGraphHash:
                            self.estimators_num_of_table[t]["buckets_used"].add(testGraphHash)
                            norm_ged_index[i][j]["lsh_use"] = True

                            model = self.trainable_buckets["model"][bucket_of_interest]
                            model.eval()

                            post_pred = model(data)
                            norm_ged_index[i][j]["post_prediction"].append(post_pred.item())

                            # tests.debugCompareModelIDs(self.model, model,
                            #                            "table {} and bucket {}.".format(t, testGraphHash))

                            # Save drill down info
                            drill_down_index.append(
                                {"table_number": t,
                                 "bucket_index": bucket_of_interest,
                                 "postpred": post_pred.item(),
                                 "priorpred": prior_pred.item(),
                                 "target": data["target"].item()}
                            )

                # The final posterior prediction value needs to be a simple float, and csv list is updated.
                if norm_ged_index[i][j]["lsh_use"] == True:
                    if len(norm_ged_index[i][j]["post_prediction"]) > 1:  # values are combined
                        pairsUtilizingManyTables += 1
                        norm_ged_index[i][j]["post_prediction"] = \
                                                                np.mean(norm_ged_index[i][j]["post_prediction"])

                    elif len(norm_ged_index[i][j]["post_prediction"]) == 1:
                        # single list value is turned into float
                        norm_ged_index[i][j]["post_prediction"] = norm_ged_index[i][j]["post_prediction"][0]

                    # Final information kept for csv.
                    csv_list.append([testgraph.i.item(),
                                     traingraph.i.item(),
                                     norm_ged_index[i][j]["target_denorm"],
                                     denormalize_sim_score(testgraph, traingraph,
                                                           norm_ged_index[i][j]["prior_prediction"]),
                                     denormalize_sim_score(testgraph, traingraph,
                                                           norm_ged_index[i][j]["post_prediction"])
                                     ])

                elif len(norm_ged_index[i][j]["post_prediction"]) == 0:
                    # If LSH isn't used post prediction is assigned the same value with the prior.
                    norm_ged_index[i][j]["post_prediction"] = norm_ged_index[i][j]["prior_prediction"]
                    norm_ged_index[i][j]["lsh_use"] = False
                    pairs_not_used += 1

                    # Final information kept for csv.
                    csv_list.append([testgraph.i.item(),
                                     traingraph.i.item(),
                                     norm_ged_index[i][j]["target_denorm"],
                                     denormalize_sim_score(testgraph, traingraph,
                                                           norm_ged_index[i][j]["prior_prediction"]),
                                     None
                                     ])

            # Before moving to the next test graph, stats are saved.
            prior = np.array([pr["prior_prediction"] for pr in norm_ged_index[i][:]])
            ground = np.array([pr["target"] for pr in norm_ged_index[i][:]])
            post = np.array([pr["post_prediction"] for pr in norm_ged_index[i][:]])

            scoresPRIOR[i] = F.mse_loss(torch.tensor(prior), torch.tensor(ground), reduction='none').detach().numpy()
            rho_listPRIOR.append(calculate_ranking_correlation(spearmanr, prior, ground))
            tau_listPRIOR.append(calculate_ranking_correlation(kendalltau, prior, ground))
            prec_at_10_listPRIOR.append(calculate_prec_at_k(10, prior, ground))
            prec_at_20_listPRIOR.append(calculate_prec_at_k(20, prior, ground))

            scoresPOST[i] = F.mse_loss(torch.tensor(post), torch.tensor(ground), reduction='none').detach().numpy()
            rho_listPOST.append(calculate_ranking_correlation(spearmanr, post, ground))
            tau_listPOST.append(calculate_ranking_correlation(kendalltau, post, ground))
            prec_at_10_listPOST.append(calculate_prec_at_k(10, post, ground))
            prec_at_20_listPOST.append(calculate_prec_at_k(20, post, ground))

            tq.update(len(self.training_graphs))

        """##########################################################################################################"""

        # Raw data to CSV
        utils.saveToFile(csv_list, "csv_results", param.lsh_res_path, param_type='csv')

        # After finishing the scoring, statistics are saved to be previewed.

        utils.saveToFile(self.args.dataset, "datasetName", param.temp_runfiles_path,  param_type='str')

        # 1ST: COMPARING VECTORIZED AND LOOP-BASED PRIORS

        zeroDifsCounts = [
            "Differences at prior predictions equal to 0: {} of {} ({}%)".format(zeroPredDifs,
                                             len(self.testing_graphs) * len(self.training_graphs),
                                             round(zeroPredDifs * 100 /
                                                   (len(self.testing_graphs) * len(self.training_graphs)), 1)
                                             ),
            "Differences at prior targets equal to 0: {} of {} ({}%)".format(zeroTargDifs,
                                                 len(self.testing_graphs) * len(self.training_graphs),
                                                 round(zeroTargDifs * 100 /
                                                       (len(self.testing_graphs) * len(self.training_graphs)), 1)
                                                 )
        ]

        utils.saveToFile(zeroDifsCounts, "zeroDifsCounts", param.vector_loopy_res_path)
        utils.saveToFile(np.abs(self.predDifsNoLSH), "AbsolutePiorDifs", param.vector_loopy_res_path)

        # histogram and heatmap to be printed

        # 2ND: COMPARING PRIOR AND POSTERIOR RESULTS
        # Note: Paper metrics are calculated on the entire dataset, while the rest only on the utilized pairs.

        # PRIOR stats
        self.model_errorPRIOR = np.mean(scoresPRIOR)
        self.rhoPRIOR = np.mean(rho_listPRIOR)
        self.tauPRIOR = np.mean(tau_listPRIOR)
        self.prec_at_10PRIOR = np.mean(prec_at_10_listPRIOR)
        self.prec_at_20PRIOR = np.mean(prec_at_20_listPRIOR)

        # POSTERIOR stats
        self.model_errorPOST = np.mean(scoresPOST)
        self.rhoPOST = np.mean(rho_listPOST)
        self.tauPOST = np.mean(tau_listPOST)
        self.prec_at_10POST = np.mean(prec_at_10_listPOST)
        self.prec_at_20POST = np.mean(prec_at_20_listPOST)

        utils.saveToFile(self.model_errorPRIOR, "global_prior_mse", param.global_res_path)
        utils.saveToFile(self.rhoPRIOR,         "global_prior_rho", param.global_res_path)
        utils.saveToFile(self.tauPRIOR,         "global_prior_tau", param.global_res_path)
        utils.saveToFile(self.prec_at_10PRIOR,  "global_prior_p@10", param.global_res_path)
        utils.saveToFile(self.prec_at_20PRIOR,  "global_prior_p@20", param.global_res_path)

        utils.saveToFile(self.model_errorPOST,  "global_post_mse", param.global_res_path)
        utils.saveToFile(self.rhoPOST,          "global_post_rho", param.global_res_path)
        utils.saveToFile(self.tauPOST,          "global_post_tau", param.global_res_path)
        utils.saveToFile(self.prec_at_10POST,   "global_post_p@10", param.global_res_path)
        utils.saveToFile(self.prec_at_20POST,   "global_post_p@20", param.global_res_path)

        # These to be printed

        # This is saved for the ERROR stat among other reasons
        utils.saveToFile(norm_ged_index, "global_norm_ged_lsh_index", param.global_res_path)

        # 3RD: LSH UTILIZATION COUNTS
        # Pairs in LSH

        num_of_pairs = len(self.testing_graphs) * len(self.training_graphs)
        used_pairs = num_of_pairs - pairs_not_used
        pairspercent1 = round(used_pairs * 100 / num_of_pairs, 1)
        pairspercent2 = round(pairsUtilizingManyTables * 100 / used_pairs, 1)
        utilizationCounts = [
            "\nLSH Usage (pairs): {} of {} ({}%)".format(used_pairs, num_of_pairs, pairspercent1),
            "Pairs that utilize more than 1 LSH table: {} of {} ({}%)".format(pairsUtilizingManyTables,
                                                                              used_pairs,
                                                                              pairspercent2),
            "\n"
        ]

        # Buckets in LSH
        for num, ith_table in enumerate(self.lsh.hash_tables):
            utilizationCounts.append("\nTable no. {}:".format(num))
            utilizationCounts.append("Total buckets: {}".format(len(ith_table.storage)))
            utilizationCounts.append("Trainable buckets: {}"
                                     .format(self.estimators_num_of_table[num]["trainable_buckets"]))
            utilizationCounts.append("Utilized buckets: {}"
                                     .format(len(self.estimators_num_of_table[num]["buckets_used"])))

        utils.saveToFile(utilizationCounts, "utilization_counts", param.lsh_res_path)

        # 4TH: INFO FOR THE DRILL DOWN IN BUCKETS PART
        tr_buckets = self.trainable_buckets
        del tr_buckets["model"]  # Removed cause unused - to supress a torch future warning for pickle save deprecation
        utils.saveToFile(tr_buckets, "trainable_buckets_dict", param.lsh_res_path,
                         param_type='dict')
        utils.saveToFile(drill_down_index, "drill_down_index", param.lsh_res_path)

