import utils, networkx, numpy as np
from torch_geometric.data import Batch

def testGEDcalclulation(trainer):
    #I made an example pair of graphs to work with
    _x  = 1
    x = trainer.training_graphs[_x]
    train5 = utils.PyG2NetworkxG(x, aids=True if trainer.args.dataset=='AIDS700nef' else False)

    _x1 = 1
    x1 = trainer.testing_graphs[_x1]
    test6 = utils.PyG2NetworkxG(x1, aids=True if trainer.args.dataset=='AIDS700nef' else False)

    utils.draw_weighted_nodes("skata.png", x, trainer)
    utils.draw_graphs({x, x1}, aids=True)
    z = utils.calculate_ged_NX(train5, test6)
    utils.print2graphsNX(train5, test6, "train {}".format(_x), "test {}".format(_x1), showGED=True, nxGED=z)
    return None


def testNetworkX(trainer):
    # I create a pair of graphs from scratch to check nx ged functions

    # Create 2 graphs
    z1 = networkx.Graph()
    z2 = networkx.Graph()

    z1.add_nodes_from([(0, {'name': 0}),
                       (1, {'name': 1}),
                       (2, {'name': 2}),
                       (3, {'name': 3})])

    z1.add_edges_from([(0, 1), (0, 2), (0, 3)])

    z2.add_nodes_from([(0, {'name': 0}),
                       (1, {'name': 1}),
                       (2, {'name': 2})])

    z2.add_edges_from([(0, 1), (0, 2)])

    utils.print2graphsNX(z1, z2, "z1", "z2", showGED=True)

    trainer.presentGraphsWithGEDs(entireDataset=False)
    return None

def debugVectorizedTesting(self, graph):
    """
    At every test graph cycle, the vectorized testing is re-done here.
    """
    # source batch is a batch with 1 test graph repeated multiple times
    source_batch = Batch.from_data_list([graph] * len(self.training_graphs))
    # target batch is a batch with every training graph
    target_batch = Batch.from_data_list(self.training_graphs)

    dataTMP = self.transform((source_batch, target_batch))
    norm_ground_truth_testgraphTMP = dataTMP["target"]
    norm_ground_truth_testgraphTMP = norm_ground_truth_testgraphTMP.detach().numpy()

    self.model.eval()
    predictionTMP = self.model(dataTMP)
    predictionTMP = predictionTMP.detach().numpy()

def debugModelsIDs(self):
    # Print the unique python object ids for the models
    for i in self.trainable_buckets["model"]:
        print(id(i))

def debugModelStateDict(self, model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.parameters():
        print(param_tensor)

def debugCompareModelIDs(globalModel, specificModel, bucket=None):
    # Print the unique python object ids for the models
    print(bucket)
    print("Self model id: {}".format(id(globalModel)))
    print("New model id: {}".format(id(specificModel)))


def verifyGedNormalization(trainer):
    denorm_ground_truth = np.zeros((len(trainer.testing_graphs), len(trainer.training_graphs)))
    data_ged_part = np.zeros((len(trainer.testing_graphs), len(trainer.training_graphs)))
    denorm_prediction_mat = np.zeros((len(trainer.testing_graphs), len(trainer.training_graphs)))

    for i in range(len(trainer.testing_graphs)):
        g1_geom = trainer.testing_graphs[i]
        g1 = utils.PyG2NetworkxG(g1_geom)

        for j in range(len(trainer.training_graphs)):
            g2_geom = trainer.training_graphs[j]
            g2 = utils.PyG2NetworkxG(g2_geom)

            GT = trainer.norm_ground_truth[i][j]
            PR = trainer.norm_prediction_mat[i][j]

            trainer.denorm_ground_truth[i][j] = utils.denormalize_sim_score(g1_geom, g2_geom, GT)
            trainer.denorm_prediction_mat[i][j] = utils.denormalize_sim_score(g1_geom, g2_geom, PR)

            data_ged_part[i][j] = trainer.ged_matrix[i + trainer.training_graphs.__len__()][j]

    return denorm_ground_truth, data_ged_part, denorm_prediction_mat


def testName(input, real):
    if input != real:
        print("Wrong input")
        import sys
        sys.exit()
