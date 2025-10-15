import argparse


class ArgumentHelper:
    @staticmethod
    def parse_main_script():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--epoch",
            type=int,
            help="Number of epochs to use for training the model",
            default=10
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            help="Number of epochs to use for training the model",
            default=64
        )

        parser.add_argument(
            "--testing",
            help="Use a small subset of the dataset for debugging",
            action="store_true"
        )

        return parser.parse_args()