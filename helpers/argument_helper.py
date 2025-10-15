import argparse


class ArgumentHelper:
    @staticmethod
    def parse_main_script():
        parser = argparse.ArgumentParser()
        parser.add_argument("--testing", help="It will not use the full dataset, just to make sure everything is going all right",
                            action='store_true')
        return parser.parse_args()