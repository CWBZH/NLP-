import pickle
from typing import Iterable, List, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class IntentClassifier:
    """Intent classifier based on TF-IDF character features and logistic regression."""

    NAVIGATION_LIKE_INTENTS = {"navigation", "add_waypoint", "query_route"}

    def __init__(self, auto_train: bool = True):
        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3))),
                ("clf", LogisticRegression(max_iter=1000, random_state=7)),
            ]
        )
        self._is_fitted = False
        if auto_train:
            self.fit(self.default_training_samples())

    def fit(self, samples: Iterable[Tuple[str, str]]):
        texts: List[str] = []
        labels: List[str] = []
        for text, label in samples:
            texts.append(text)
            labels.append(label)

        if len(set(labels)) < 2:
            raise ValueError("IntentClassifier requires at least two intent labels.")

        self.model.fit(texts, labels)
        self._is_fitted = True
        return self

    def predict(self, text: str) -> str:
        if not self._is_fitted:
            raise RuntimeError("IntentClassifier is not fitted.")
        return self.model.predict([text])[0]

    def is_navigation_like(self, intent: str) -> bool:
        return intent in self.NAVIGATION_LIKE_INTENTS

    def save(self, path: str) -> None:
        with open(path, "wb") as file:
            pickle.dump({"model": self.model, "is_fitted": self._is_fitted}, file)

    @classmethod
    def load(cls, path: str):
        classifier = cls(auto_train=False)
        with open(path, "rb") as file:
            payload = pickle.load(file)
        classifier.model = payload["model"]
        classifier._is_fitted = payload["is_fitted"]
        return classifier

    @staticmethod
    def default_training_samples() -> Sequence[Tuple[str, str]]:
        return [
            ("从蓝色点到绿色点", "navigation"),
            ("我要从蓝点去绿点", "navigation"),
            ("帮我规划从紫色点到绿色点的路线", "navigation"),
            ("去绿色点，从蓝色点出发", "navigation"),
            ("从蓝色点出发", "navigation"),
            ("到绿色点", "navigation"),
            ("我想从蓝色点走到绿色点", "navigation"),
            ("能不能帮我从红点导航到紫点", "navigation"),
            ("帮我找一条蓝点到绿点的路", "navigation"),
            ("我在青色点，想去红色点", "navigation"),
            ("从橙色点开始，最后去蓝色点", "navigation"),
            ("目标是绿色点，我从蓝色点出发", "navigation"),
            ("从蓝色点到绿色点，不经过紫色点", "navigation"),
            ("我要从蓝色点出发，经过紫，到蓝色，不要经过黄色", "navigation"),
            ("从红点到蓝点，途径橙点，避开黄色点", "navigation"),
            ("从青点到紫点，先经过绿点，不要路过橙点", "navigation"),
            ("从蓝到红，绕开黄，经过紫", "navigation"),
            ("从橙到青，不走红，经过绿", "navigation"),
            ("蓝点出发经过黄点最后到绿点", "navigation"),
            ("再经过黄色点", "add_waypoint"),
            ("帮我加一个青色点作为途经点", "add_waypoint"),
            ("途径黄和紫", "add_waypoint"),
            ("现在这条路线怎么走", "query_route"),
            ("查询当前路线", "query_route"),
            ("路线是什么", "query_route"),
            ("今天天气怎么样", "unknown"),
            ("讲个笑话", "unknown"),
            ("北京现在几点", "unknown"),
            ("红色是什么颜色", "unknown"),
            ("打开音乐", "unknown"),
        ]
