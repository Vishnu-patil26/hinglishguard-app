"""The classical (TF-IDF) feature extractor. Lives in the package so a saved model can be loaded anywhere."""
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .lid import int32_csr


class Tfidf:
    """word 1-2 grams + character 2-5 grams (char n-grams soak up Hinglish's free spelling)."""

    def __init__(self) -> None:
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=120000, sublinear_tf=True)

    def fit_transform(self, texts):
        return int32_csr(hstack([self.w.fit_transform(texts), self.c.fit_transform(texts)]))

    def transform(self, texts):
        return int32_csr(hstack([self.w.transform(texts), self.c.transform(texts)]))
