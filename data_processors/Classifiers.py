from abc import abstractmethod, ABC

from transformers import pipeline
import numpy as np
import torch

DEVICE = "cuda" if torch.torch.cuda.is_available() else "cpu"


class Classifer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._categories = set()

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, options):
        self._categories = set(options)

    @categories.deleter
    def categories(self, category):
        self._categories.remove(category)

    @abstractmethod
    def classify(self, text):
        pass


class ZeroShotClassifer(Classifer):
    def __init__(self) -> None:
        super().__init__()
        self.pipe = pipeline(model="facebook/bart-large-mnli", device=DEVICE)

    def classify(self, text):
        answer = self.pipe(text, candidate_labels=list(self._categories))

        i = np.argmax(answer["scores"])
        return answer["labels"][i]
