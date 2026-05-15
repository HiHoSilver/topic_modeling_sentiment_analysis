from typing import Self
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import umap
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io_utils import load_df


class TopicSentimentPipeline:
    def __init__(
        self,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        n_neighbors=15,
        n_components=5,
        min_cluster_size=10,
        cluster_method="hdbscan",
        cluster_selection_method="eom",
        kmeans_k=10,
        n_top_terms=10,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            random_state=42,
        )
        self.cluster_method = cluster_method
        self.kmeans_k = kmeans_k
        self.cluster_selection_method = cluster_selection_method
        self.min_cluster_size = min_cluster_size
        self.n_top_terms = n_top_terms

        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=5000, ngram_range=(1, 2)
        )
        self.sia = SentimentIntensityAnalyzer()

        self.topic_terms_ = {}
        self.clusterer = None

    def _extract_docs(self, data, column):
        if isinstance(data, pd.DataFrame):
            return data[column].astype(str).tolist()
        return data

    def embed(self, docs):
        return self.model.encode(docs, show_progress_bar=True)

    def reduce(self, embeddings):
        return self.reducer.fit_transform(embeddings)

    def cluster(self, reduced):
        if self.cluster_method == "kmeans":
            self.clusterer = KMeans(
                n_clusters=self.kmeans_k,
                random_state=42,
                n_init="auto",
            )
            return self.clusterer.fit_predict(reduced)

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
        )
        return self.clusterer.fit_predict(reduced)

    def extract_topic_terms(self, docs, labels):
        df = pd.DataFrame({"doc": docs, "topic": labels})
        df = df[df.topic != -1].reset_index(drop=True)

        if df.empty:
            return {}

        X = self.vectorizer.fit_transform(df["doc"])
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        topic_terms = {}
        for topic_id, group in df.groupby("topic"):
            idx = group.index.to_list()
            tfidf_scores = X[idx].mean(axis=0).A1
            top_idx = np.argsort(tfidf_scores)[::-1][: self.n_top_terms]
            topic_terms[topic_id] = feature_names[top_idx].tolist()

        self.topic_terms_ = topic_terms
        return topic_terms

    def score_sentiment(self, docs):
        return [self.sia.polarity_scores(d)["compound"] for d in docs]

    def aggregate_sentiment(self, labels, sentiment):
        df = pd.DataFrame({"topic": labels, "sentiment": sentiment})
        df = df[df.topic != -1]
        return df.groupby("topic")["sentiment"].mean().to_dict()

    def fit(self, data, column=None):
        docs = self._extract_docs(data, column)
        embeddings = self.embed(docs)
        reduced = self.reduce(embeddings)
        labels = self.cluster(reduced)
        topic_terms = self.extract_topic_terms(docs, labels)
        sentiment = self.score_sentiment(docs)
        topic_sentiment = self.aggregate_sentiment(labels, sentiment)

        return {
            "labels": labels,
            "topic_terms": topic_terms,
            "topic_sentiment": topic_sentiment,
            "sentiment_scores": sentiment,
        }

    def transform(self, data, column=None, reassign_noise=False):
        docs = self._extract_docs(data, column)

        embeddings = self.embed(docs)
        reduced = self.reduce(embeddings)
        raw_labels = self.cluster(reduced)

        # Optional noise reassignment BEFORE renumbering
        if reassign_noise:
            df_temp = pd.DataFrame({"topic": raw_labels})
            reassigner = NoiseReassigner(df_temp, embeddings)
            df_temp = reassigner.reassign()
            raw_labels = df_temp["topic"].values

        # Now renumber AFTER reassignment
        labels = np.where(raw_labels == -1, -1, raw_labels + 1)

        sentiment = self.score_sentiment(docs)
        topic_terms = self.extract_topic_terms(docs, labels)

        df = pd.DataFrame(
            {
                "text": docs,
                "topic": labels,
                "sentiment": sentiment,
            }
        )

        df["topic_keywords"] = df["topic"].apply(lambda t: topic_terms.get(t, []))
        df["topic_name"] = df["topic_keywords"].apply(
            lambda kws: ", ".join(kws[:3]) if kws else "Noise / Unassigned"
        )

        return df


class NoiseReassigner:
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings

    def reassign(self):
        df = self.df.copy()
        embeddings = self.embeddings

        # Identify noise points
        noise_mask = df["topic"].astype(int) == -1
        noise_embeddings = embeddings[noise_mask]

        # If no noise, return original df
        if len(noise_embeddings) == 0:
            return df

        # Identify real clusters
        real_topics = df[df["topic"] != -1]["topic"].unique()
        real_topics.sort()

        # Compute centroids of real clusters
        centroids = []
        for topic in real_topics:
            topic_mask = df["topic"] == topic
            topic_embeddings = embeddings[topic_mask]
            centroids.append(topic_embeddings.mean(axis=0))

        centroids = np.vstack(centroids)

        # Run KMeans on noise points with k = number of real clusters
        kmeans = KMeans(n_clusters=len(real_topics), random_state=42, n_init="auto")
        kmeans.fit(noise_embeddings)

        # Assign each noise point to nearest real cluster centroid
        noise_to_cluster = []
        for i, emb in enumerate(noise_embeddings):
            sims = cosine_similarity([emb], centroids).flatten()
            best_cluster_idx = sims.argmax()
            noise_to_cluster.append(real_topics[best_cluster_idx])

        # Update df
        df.loc[noise_mask, "topic"] = pd.Series(
            noise_to_cluster, index=df.index[noise_mask]
        )

        return df


class RepresentativeResponseExtractor:
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings
        self.reps = {}

    def get_representative_responses(self, top_k=3):
        df = self.df
        embeddings = self.embeddings

        for topic in df["topic"].unique():
            topic_mask = df["topic"] == topic
            topic_df = df[topic_mask]
            topic_embeddings = embeddings[topic_mask]

            if len(topic_df) == 0:
                continue

            centroid = topic_embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(topic_embeddings, centroid).flatten()
            top_idx = sims.argsort()[::-1][:top_k]

            self.reps[topic] = topic_df.iloc[top_idx]["text"].tolist()

        return self.reps

    def export_to_excel(self, reps, output_path="representative_responses.xlsx"):
        rows = []

        for topic, texts in reps.items():
            topic_name = self.df[self.df["topic"] == topic]["topic_name"].iloc[0]

            for t in texts:
                rows.append(
                    {
                        "topic": topic,
                        "topic_name": topic_name,
                        "representative_response": t,
                    }
                )

        rep_df = pd.DataFrame(rows)
        noise_df = rep_df.loc[rep_df["topic"] == -1]
        non_noise_df = rep_df.loc[rep_df["topic"] > -1].sort_values("topic")

        sorted_df = pd.concat([non_noise_df, noise_df])

        sorted_df.to_excel(output_path, index=False)

        return self


class VisualizationGenerator:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def topic_distribution(self) -> Self:
        plt.figure(figsize=(10, 5))
        topic_order = self.df["topic_name"].value_counts().index

        sns.countplot(
            data=self.df, y="topic_name", order=topic_order, palette="viridis"
        )

        plt.title("Topic Distribution")
        plt.xlabel("Number of Responses")
        plt.ylabel("Topic")
        plt.tight_layout()
        plt.savefig("topic_distribution.png")
        return self

    def sentiment_distribution(self) -> Self:
        plt.figure(figsize=(12, 6))
        order = self.df.groupby("topic_name")["sentiment"].mean().sort_values().index

        sns.boxplot(
            data=self.df,
            x="sentiment",
            y="topic_name",
            order=order,
            palette="coolwarm",
            showmeans=True,
            meanline=True,
            meanprops={"color": "black", "linewidth": 1},
        )

        plt.title("Sentiment Distribution by Topic")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Topic")
        plt.tight_layout()
        plt.savefig("sentiment_distribution.png")
        return self

    def umap_embedding(self, embeddings, output_path="umap_embedding.png") -> Self:
        """
        Visualize the 2D UMAP embedding colored by topic.
        Requires: embeddings (2D array) from pipeline.embed()
        """
        # Reduce to 2D for visualization
        reducer = umap.UMAP(
            n_neighbors=30, n_components=2, metric="cosine", random_state=42
        )
        emb_2d = reducer.fit_transform(embeddings)

        df_plot = pd.DataFrame(
            {
                "x": emb_2d[:, 0],
                "y": emb_2d[:, 1],
                "topic": self.df["topic"],
                "topic_name": self.df["topic_name"],
            }
        )

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df_plot,
            x="x",
            y="y",
            hue="topic_name",
            palette="tab20",
            s=40,
            alpha=0.8,
            edgecolor="none",
        )

        plt.title("UMAP Embedding of Responses by Topic")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)

        return self

    def umap_embedding_3d_plotly(
        self, embeddings, output_path="umap_embedding_3d.html"
    ) -> Self:
        """
        Create an interactive 3D UMAP embedding plot using Plotly.
        Saves to an HTML file that can be opened in any browser.
        """
        reducer = umap.UMAP(
            n_neighbors=30, n_components=3, metric="cosine", random_state=42
        )
        emb_3d = reducer.fit_transform(embeddings)

        df_plot = pd.DataFrame(
            {
                "x": emb_3d[:, 0],
                "y": emb_3d[:, 1],
                "z": emb_3d[:, 2],
                "topic": self.df["topic"],
                "topic_name": self.df["topic_name"],
                "text": self.df["text"],
            }
        )

        fig = px.scatter_3d(
            df_plot,
            x="x",
            y="y",
            z="z",
            color="topic_name",
            hover_data={"text": True, "topic_name": True},
            opacity=0.85,
            title="3D UMAP Embedding of Responses by Topic",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )

        fig.update_traces(marker=dict(size=5))
        fig.update_layout(legend=dict(x=1.05, y=1, bgcolor="rgba(255,255,255,0.7)"))

        fig.write_html(output_path)

        return self


def main() -> None:
    start_time: float = time.perf_counter()


    # Load and shape data
    df = load_df(
        "wfd_rfi.xlsx",
        usecols=[
            "Describe the greatest opportunities or challenges to creating flexible and affordable training programs (for technicians, practitioners, researchers, students, etc.) needed to build an inclusive, well-paid, domestic workforce in emerging technology careers. (maximum 600 words):"
        ],
    )
    df = df.dropna(axis=0, how="any", subset=None, inplace=False)
    df = df.rename(
        columns={
            "Describe the greatest opportunities or challenges to creating flexible and affordable training programs (for technicians, practitioners, researchers, students, etc.) needed to build an inclusive, well-paid, domestic workforce in emerging technology careers. (maximum 600 words):": "text"
        }
    )

    # 1. Run pipeline
    pipeline = TopicSentimentPipeline(
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        cluster_method="hdbscan",
        min_cluster_size=3,
        cluster_selection_method="eom",
        n_components=15,
        n_neighbors=50,
    )

    pipeline.fit(df, column="text")
    df_out = pipeline.transform(df, column="text", reassign_noise=True)

    # 2. Compute embeddings
    embeddings = pipeline.embed(df_out["text"].tolist())

    # 3. Noise reassignment (NEW)
    reassigner = NoiseReassigner(df_out, embeddings)
    df_out = reassigner.reassign()

    # 4. Export corrected topic assignments to Excel
    df_out.to_excel("topic_analysis.xlsx", index=False)

    # 5. Extract representative responses
    extractor = RepresentativeResponseExtractor(df_out, embeddings)
    reps = extractor.get_representative_responses(top_k=3)
    extractor.export_to_excel(reps, "representative_responses.xlsx")

    # 6. Visualizations
    viz = VisualizationGenerator(df_out)
    viz.topic_distribution().sentiment_distribution().umap_embedding(
        embeddings
    ).umap_embedding_3d_plotly(embeddings)


    end_time: float = time.perf_counter()
    elapsed_time: float = end_time - start_time
    print(f"Runtime: {elapsed_time:.4f} seconds")


if __name__ == "__main__":
    main()
