from typing import Self
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly_express as px
import seaborn as sns
import umap
from sklearn.metrics.pairwise import cosine_similarity


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
        plt.savefig("outputs/topic_distribution.png")
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
        plt.savefig("outputs/sentiment_distribution.png")
        return self

    def umap_embedding(
        self, embeddings, output_path="outputs/umap_embedding.png"
    ) -> Self:
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
        self, embeddings, output_path="outputs/umap_embedding_3d.html"
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

        # Short preview for hover
        df_plot["hover_short"] = df_plot["text"].str.slice(0, 200) + "..."

        # Full text for click events
        df_plot["full_text"] = df_plot["text"]

        fig = px.scatter_3d(
            df_plot,
            x="x",
            y="y",
            z="z",
            color="topic_name",
            opacity=0.85,
            title="3D UMAP Embedding of Responses by Topic",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )

        # Attach full text + short text to each point
        fig.update_traces(
            customdata=df_plot[["hover_short", "full_text"]],
            hovertemplate="%{customdata[0]}<extra></extra>",
            marker=dict(size=5),
        )

        fig.update_layout(legend=dict(x=1.05, y=1, bgcolor="rgba(255,255,255,0.7)"))
        fig.write_html(output_path)
        return self

    def topic_similarity_matrix(
        self, embeddings, output_path="outputs/topic_similarity_matrix.png"
    ):
        df = self.df.copy()

        # Compute centroids
        centroids = {}
        for topic in df["topic"].unique():
            mask = df["topic"] == topic
            centroids[topic] = embeddings[mask].mean(axis=0)

        # Build similarity matrix
        topics = sorted(centroids.keys())
        matrix = np.zeros((len(topics), len(topics)))

        for i, ti in enumerate(topics):
            for j, tj in enumerate(topics):
                matrix[i, j] = cosine_similarity([centroids[ti]], [centroids[tj]])[0][0]

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            xticklabels=topics,
            yticklabels=topics,
            cmap="viridis",
            annot=True,
            fmt=".2f",
        )
        plt.title("Topic Similarity Matrix (Cosine Similarity)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        return self

    def topic_embedding_density(
        self, embeddings, output_path="outputs/topic_density.png"
    ):
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

        plt.figure(figsize=(12, 10))
        sns.kdeplot(
            data=df_plot,
            x="x",
            y="y",
            hue="topic_name",
            fill=True,
            alpha=0.4,
            thresh=0.05,
            levels=20,
        )

        plt.title("Topic Embedding Density Plot")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        return self

    def umap_cluster_confidence(
        self, embeddings, clusterer, output_path="outputs/umap_confidence.png"
    ):
        reducer = umap.UMAP(
            n_neighbors=30, n_components=2, metric="cosine", random_state=42
        )
        emb_2d = reducer.fit_transform(embeddings)

        df_plot = pd.DataFrame(
            {
                "x": emb_2d[:, 0],
                "y": emb_2d[:, 1],
                "topic": self.df["topic"],
                "confidence": clusterer.probabilities_,
            }
        )

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df_plot,
            x="x",
            y="y",
            hue="confidence",
            palette="viridis",
            s=40,
            alpha=0.9,
        )

        plt.title("UMAP Cluster Confidence (HDBSCAN Probabilities)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        return self

    def topic_keyword_overlap_matrix(
        self, topic_terms, output_path="outputs/topic_keyword_overlap.png"
    ):
        # Filter out topics with no keywords
        topic_terms = {t: kws for t, kws in topic_terms.items() if len(kws) > 0}

        if len(topic_terms) == 0:
            print("No topics have keywords — skipping keyword overlap matrix.")
            return self

        topics = sorted(topic_terms.keys())
        n = len(topics)

        matrix = np.zeros((n, n))

        for i, ti in enumerate(topics):
            set_i = set(topic_terms[ti])
            for j, tj in enumerate(topics):
                set_j = set(topic_terms[tj])

                # Safe Jaccard similarity
                union = set_i | set_j
                if len(union) == 0:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = len(set_i & set_j) / len(union)

        # If matrix is all zeros, seaborn will fail — handle gracefully
        if np.all(matrix == 0):
            print("Keyword overlap matrix is all zeros — no overlaps to visualize.")
            return self

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            xticklabels=topics,
            yticklabels=topics,
            cmap="magma_r",
            annot=True,
            fmt=".2f",
        )

        plt.title("Topic Keyword Overlap (Jaccard Similarity)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        return self
