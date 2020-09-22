import torch


import utils, time, lsh
from trainer import SimGNNTrainer
from plottingFunctions import showLSHTablesDistributions
import param_parser as param
from mainForResults import mainResults

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SimGNN model.
    """
    answer = input("Are you sure you want to run the pipeline? Previous temp result files will be lost! (type yes)")
    if answer != 'yes':
        import sys
        sys.exit()

    # Parse and Print parameters
    args = param.parameter_parser()
    utils.tab_printer(args)
    dataset_name = args.dataset
    param.initGlobals(dataset_name)
    utils.clearDataFolder(param.temp_runfiles_path)

    # Initiate SimGnn Trainer
    trainer = SimGNNTrainer(args)

    use_pretrained = False  # To determine whether a pre-trained model will be used.
    path = "../saved_model_state/general_simgnn_embSize{}.pt".format(trainer.embeddings_size)

    #TRAIN
    if use_pretrained is True:
        print("Pre-trained mode: load an already fit state instead of training.")

        checkpoint = torch.load(path)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        trainer.fit()

        # I save the trained state to disk.
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict()
        }, path)

    #TEST
    trainer.globalScore()  # Vectorized scoring - can be skipped when using LSH

    if args.use_lsh:
        tuneLSH = False  # To determine if LSH module will be utilized or examined.

        if tuneLSH:
            showLSHTablesDistributions(trainer, dataset_name, saveToFile=True)
            import sys
            sys.exit()
        else:
            lsh.makeLSHIndex(trainer)

            lsh.createBucketsWithNetworks(trainer, path)

            trainer.lshScore()

    if args.notify:
        import os, sys
        if sys.platform == 'linux':
            os.system( 'notify-send SimGNN "Program is finished.')
        elif sys.platform == 'posix':
            os.system("""osascript -e 'display notification "SimGNN" with title "Program is finished."'
                      """)
        else:
            raise NotImplementedError('No support for this OS.')

    mainResults(dataset_name, args.use_lsh)




if __name__ == "__main__":
    main()
