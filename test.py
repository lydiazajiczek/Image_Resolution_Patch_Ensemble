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
                        help="Name of dataset classifier was trained on")
    parser.add_argument('--degradation', type=str, default='',
                        help="Degradation of dataset model was trained on, default none/original images")
    parser.add_argument('--test-degradation', type=str, default='',
                        help="Degradation of test dataset to predict on, default none/original images")
    parser.add_argument('--classifier_type', type=str, required=True,
                        help="Type of classifier for prediction: patch/whole_image")
    parser.add_argument('--base_path', type=str, default=base_path,
                        help="Base path where all datasets are located")

    # Hyper parameters below (must match original trained model)
    parser.add_argument('--num_fc_layers', type=int, default=128,
                        help="Number of neurons in fully connected layer in top classifier (default 128)")
    parser.add_argument('--model_name_suffix', type=str, default='',
                        help="Extra string that was appended to model name for testing (default none)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Threshold for positive prediction (must be between 0 and 1)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for testing (default 32)")

    return parser


if __name__ == '__main__':
    args = arg_parser().parse_args()
    patch_ensemble = PatchEnsemble(dataset=args.dataset, degradation=args.degradation,
                                   classifier_type=args.classifier_type, base_path=args.base_path,
                                   num_fc_layers=args.num_fc_layers, model_name_suffix=args.model_name_suffix)

    patch_ensemble.test(test_degradation=args.test_degradation, threshold=args.threshold, batch_size=args.batch_size)
