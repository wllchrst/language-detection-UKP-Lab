from abc import  ABC, abstractmethod

class BasePipeline(ABC):
    @abstractmethod
    def predict(self, input: str) -> str:
        pass