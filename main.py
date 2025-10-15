from pipeline.lang_identification_pipeline import LangIdentificationPipeline
from helpers.argument_helper import ArgumentHelper

def main():
    arguments = ArgumentHelper.parse_main_script()
    LangIdentificationPipeline(testing=arguments.testing)

if __name__ == "__main__":
    main()