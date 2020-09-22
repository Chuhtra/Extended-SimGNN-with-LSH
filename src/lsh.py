from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from simgnn import SimGNN
from utils import tab_printer
import torch


class ListDataset(InMemoryDataset):
    # TODO: apla prosthese root gia na swzetai eksw apo to src.
    def __init__(self, data_list, transform=None):
        root = "../BucketDatasetDir/"
        super(ListDataset, self).__init__(root, transform)
        self.data_list = data_list

    def get(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)


def makeLSHIndex(trainer, hash_size=10, num_of_tables=6, min_bucket_size=25):
    # First, extract all train graph embeddings
    trainer.embeddings_dictionary = {"trainGraphs": trainer.training_graphs}
    pooledF2, _, _ = trainer.model.get_embedding(Batch.from_data_list(trainer.training_graphs))
    trainer.embeddings_dictionary["trainGraphsEmbeddings"] = pooledF2.detach().numpy()

    # Initialize LSH parameters
    from lshashpy3 import LSHash

    k = hash_size  # hash size - number of hyperplanes
    d = trainer.embeddings_size  # Dimension of input feature vector
    L = num_of_tables  # number of tables - how many different sets of hyperplanes
    tab_printer({'hash_size': k, 'embedding_size': d, 'number_of_tables': L})
    trainer.lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)

    # Extracted embeddings are indexed, along with graph Data object as extra data.
    print("Indexing graph embeddings with LSH...")
    for i in tqdm(range(0, trainer.training_graphs.__len__() - 1)):
        graph = trainer.embeddings_dictionary["trainGraphs"][i]
        embedding = trainer.embeddings_dictionary["trainGraphsEmbeddings"][i]

        trainer.lsh.index(embedding, extra_data=graph)

    trainer.lsh.min_bucket_size = min_bucket_size


def createBucketsWithNetworks(trainer, path):
    """
    For every table's bucket that is big enough, create a new network and train it with this bucket.
    """
    # Here a list with the model and the corresponding "address" (table/bucket) is indexed.
    trainer.trainable_buckets = {"model": [], "table": [], "bucketName": []}

    # this list will hold dictionaries with numbers of trainable buckets and utilized buckets.
    trainer.estimators_num_of_table = []

    for num, ith_table in enumerate(trainer.lsh.hash_tables):
        print("\nPost-LSH training for buckets in table no. {} of {}...".format(num + 1, trainer.lsh.num_hashtables))
        # below a set is used to avoid duplicate counting
        table_estimators = {"buckets_used": set(), "trainable_buckets": 0}
        for jth_hashcode in ith_table.storage.items():
            graphs_in_bucket = [i[1] for i in jth_hashcode[1]]

            if len(graphs_in_bucket) >= trainer.lsh.min_bucket_size:  # If it's a trainable bucket.
                table_estimators["trainable_buckets"] += 1
                # Turn the bucket into a Dataset
                x = ListDataset(data_list=graphs_in_bucket)

                # Create a model for this bucket.
                bucket_model = SimGNN(trainer.args, trainer.number_of_node_labels, trainer.number_of_edge_labels)
                bucket_optimizer = torch.optim.Adam(bucket_model.parameters(), lr=trainer.args.learning_rate,
                                                    weight_decay=trainer.args.weight_decay)
                checkpoint = torch.load(path)
                bucket_model.load_state_dict(checkpoint['model_state_dict'])
                bucket_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Save the model's address.
                trainer.trainable_buckets["table"].append(num)
                trainer.trainable_buckets["bucketName"].append(jth_hashcode[0])

                # Train the model and save it to the index.
                trainer.fit(lsh_bucket=x, modelToUse=(bucket_model, bucket_optimizer))
                trainer.trainable_buckets["model"].append(bucket_model)

        # Add the estimators' counts to the list before moving to the next table.
        trainer.estimators_num_of_table.append(table_estimators)

    pass
    print("\nPost-LSH training completed.")
