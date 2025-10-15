from pipeline.lang_identification_pipeline import LangIdentificationPipeline
from helpers.argument_helper import ArgumentHelper

def main():
    arguments = ArgumentHelper.parse_main_script()
    print(f'Script run with this arguments:\n{arguments}')
    LangIdentificationPipeline(
        testing=arguments.testing,
        batch_size=arguments.batch_size,
        epoch=arguments.epoch
    )

if __name__ == "__main__":
    main()