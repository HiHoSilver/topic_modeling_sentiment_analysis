from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from collections import Counter
import numpy as np
import pandas as pd


class Validator:
    """
    Validation class for assessing clustering and topic quality.
    Requires:
        df: DataFrame with columns ["text", "topic", "topic_keywords"]
        embeddings: np.ndarray of shape (n_samples, embedding_dim)
    """

    def __init__(self, df: pd.DataFrame, embeddings=None):
        self.df = df
        self.embeddings = embeddings

    # ---------------------------------------------------------
    # Topic Coherence
    # ---------------------------------------------------------
    def _prepare_corpus(self):
        docs_tokens = (
            self.df["text"].fillna("").astype(str).str.lower().str.split().tolist()
        )
        dictionary = Dictionary(docs_tokens)
        return docs_tokens, dictionary

    def _topic_word_lists(self, top_n=10):
        topic_word_lists = {}
        for topic_id, group in self.df.groupby("topic"):
            if topic_id == -1:
                continue

            all_terms = [t for kws in group["topic_keywords"] for t in kws]
            if not all_terms:
                continue

            top_terms = [w for w, _ in Counter(all_terms).most_common(top_n)]
            topic_word_lists[topic_id] = top_terms

        return topic_word_lists

    def topic_coherence(self, top_n=10, measure="c_v"):
        docs_tokens, dictionary = self._prepare_corpus()
        topic_word_lists = self._topic_word_lists(top_n=top_n)

        topic_ids = list(topic_word_lists.keys())
        topics = list(topic_word_lists.values())

        if not topics:
            return {}, 0.0

        coherence_model = CoherenceModel(
            topics=topics,
            texts=docs_tokens,
            dictionary=dictionary,
            coherence=measure,
        )

        scores = coherence_model.get_coherence_per_topic()
        topic_coherence = dict(zip(topic_ids, scores))
        mean_coherence = float(np.mean(scores))

        return topic_coherence, mean_coherence

    # ---------------------------------------------------------
    # Cluster Size Distribution
    # ---------------------------------------------------------
    def cluster_sizes(self):
        return self.df["topic"].value_counts().to_dict()

    # ---------------------------------------------------------
    # Topic Keyword Overlap
    # ---------------------------------------------------------
    def keyword_overlap(self, top_n=10):
        topic_word_lists = self._topic_word_lists(top_n=top_n)
        overlaps = {}

        topics = list(topic_word_lists.keys())
        for i in range(len(topics)):
            for j in range(i + 1, len(topics)):
                t1, t2 = topics[i], topics[j]
                w1, w2 = set(topic_word_lists[t1]), set(topic_word_lists[t2])
                overlap = len(w1 & w2) / max(len(w1 | w2), 1)
                overlaps[(t1, t2)] = overlap

        return overlaps

    # ---------------------------------------------------------
    # Representative Response Consistency
    # ---------------------------------------------------------
    def representative_consistency(self):
        if self.embeddings is None:
            return {}

        df = self.df.copy()
        df["embedding"] = list(self.embeddings)

        consistency = {}
        for topic_id, group in df.groupby("topic"):
            if topic_id == -1 or len(group) < 2:
                continue

            embs = np.vstack(group["embedding"].values)
            centroid = embs.mean(axis=0)

            distances = np.linalg.norm(embs - centroid, axis=1)
            consistency[topic_id] = float(distances.mean())

        return consistency

    # ---------------------------------------------------------
    # Full Validation Report
    # ---------------------------------------------------------
    def report(self):
        topic_coh, mean_coh = self.topic_coherence()
        sizes = self.cluster_sizes()
        overlap = self.keyword_overlap()
        rep_consistency = self.representative_consistency()

        return {
            "mean_coherence": mean_coh,
            "topic_coherence": topic_coh,
            "cluster_sizes": sizes,
            "keyword_overlap": overlap,
            "representative_consistency": rep_consistency,
        }
