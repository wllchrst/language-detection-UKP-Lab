from pipeline.base_pipeline import BasePipeline
from datasets import Dataset
from typing import Tuple

class LangIdentificationPipeline(BasePipeline):
    """
    A pipeline for language identification specific using classifier that is trained using this class.

    Pipeline without using any training is available on other class within this folder.
    """
    def __init__(self):
        super().__init__()

    def train(self):
        pass

    def test(self):
        pass
    
    def embed_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass

    def predict(self, input: str) -> str:
        return "en"