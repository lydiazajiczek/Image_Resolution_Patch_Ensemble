from PatchEnsemble import PatchEnsemble
import argparse
import os

# base_path = 'MODIFY/TO/PATH/TO/FOLDER/CONTAINING/ALL/DATASETS'
# base_path = 'C:\\Users\\lydia\\Code\\github_repos\\Image_Quality\\'
base_path = os.getcwd()


def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


def arg_parser():
    # Defines an argument parser for the script
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, required=True,
                        help="Name of dataset to train classifier on")
    parser.add_argument('--degradation', type=str, default='',
                        help="Degradation of dataset to train on, default none/original images")
    parser.add_argument('--classifier_type', type=str, required=True,
                        help="Type of classifier to train: patch/whole_image")
    parser.add_argument('--base_path', type=str, default=base_path,
                        help="Base path where all datasets are located")
    parser.add_argument('--train_top', type=parse_boolean, default=True,
                        help="Flag whether to train top classifier on bottleneck features (default True)")
    parser.add_argument('--extract_features', type=parse_boolean, default=True,
                        help="Flag whether to extract bottleneck features or load previous (default True)")
    parser.add_argument('--fine_tune', type=parse_boolean, default=True,
                        help="Flag whether to fine tune model (default True)")
    parser.add_argument('--patch_weights', type=str, default='',
                        help="Degradation weights of previously trained patch model to load (default baseline)")
    parser.add_argument('--use_baseline_weights', type=parse_boolean, default=True,
                        help="Flag whether to use top weights from baseline model")

    # Hyper parameters below (all optional)
    parser.add_argument('--epochs', type=int, default=100,
                        help="Number of epochs to train for (default 100)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training (default 32)")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate for training (default 1e-4)")
    parser.add_argument('--beta_1', type=float, default=0.6,
                        help="Beta_1 parameter for Adam optimizer (default 0.6)")
    parser.add_argument('--beta_2', type=float, default=0.8,
                        help="Beta_2 parameter for Adam optimizer (default 0.8)")
    parser.add_argument('--num_fc_layers', type=int, default=128,
                        help="Number of neurons in fully connected layer in top classifier (default 128)")
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help="Dropout rate of top classifier (default 0.5)")
    parser.add_argument('--validation_split', type=float, default=0.25,
                        help="Split for dividing training/validation set if required (default 0.25)")
    parser.add_argument('--model_name_suffix', type=str, default='',
                        help="String to append to model name for testing (default none)")
    parser.add_argument('--vgg_layer', type=int, default=16,
                        help="First layer to unfreeze in VGG19 network for fine tuning (default 16")
    parser.add_argument('--mnet_layer', type=int, default=97,
                        help="First layer to unfreeze in MobileNetv2 network for fine tuning (default 97")
    parser.add_argument('--dnet_layer', type=int, default=480,
                        help="First layer to unfreeze in DenseNet201 network for fine tuning (default 480")

    return parser


if __name__ == '__main__':
    args = arg_parser().parse_args()
    patch_ensemble = PatchEnsemble(dataset=args.dataset, degradation=args.degradation,
                                   classifier_type=args.classifier_type, base_path=args.base_path,
                                   num_fc_layers=args.num_fc_layers, dropout_rate=args.dropout_rate,
                                   vgg_layer=args.vgg_layer, mnet_layer=args.mnet_layer, dnet_layer=args.dnet_layer,
                                   validation_split=args.validation_split, model_name_suffix=args.model_name_suffix)

    if args.train_top:
        if args.classifier_type == 'whole_image':
            raise ValueError('Bottleneck feature extraction is not compatible with random patch extraction.')
        patch_ensemble.train_top(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                 extract_features=args.extract_features)

    if args.fine_tune:
        patch_ensemble.fine_tune(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                                 patch_weights=args.patch_weights, use_baseline_weights=args.use_baseline_weights)
