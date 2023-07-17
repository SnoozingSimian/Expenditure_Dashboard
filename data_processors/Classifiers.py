from abc import abstractmethod, ABC

from transformers import pipeline
import numpy as np


class Classifer(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._categories = set()

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, **kwargs):
        for k, v in kwargs.items():
            self._categories.add(v)

    @categories.deleter
    def categories(self, category):
        self._categories.remove(category)

    @abstractmethod
    def classify(self, text):
        pass


class ZeroShotClassifer(Classifer):
    def __init__(self) -> None:
        super().__init__()
        self.pipe = pipeline(model="facebook/bart-large-mnli")

    def classify(self, text):
        answer = self.pipe(text, candidate_labels=self._categories)

        i = np.argmax(answer["scores"])
        return answer["labels"][i]
