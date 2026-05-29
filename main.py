from typing import Self
import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
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
from config import DATA_CONFIG
from io_utils import load_df
from validation import Validator


class TopicModelingPipeline:
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
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(self.embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        self.tokenizer.model_max_length = 100000
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

    def _chunk_text_token_aware(self, text, max_tokens=350, overlap=50):
        """
        Splits text into chunks based on tokenizer word-piece tokens.
        Ensures no chunk exceeds the model's max token limit (512).
        """
        # Tokenize into word pieces
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        input_ids = encoding["input_ids"]
        chunks = []
        start = 0

        while start < len(input_ids):
            end = start + max_tokens
            chunk_ids = input_ids[start:end]

            # Convert token IDs back to text
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

            start += max_tokens - overlap

        return chunks

    def _embed_chunks(self, text):
        chunks = self._chunk_text_token_aware(text)

        # Encode each chunk safely
        chunk_embeddings = self.model.encode(
            chunks, show_progress_bar=False, normalize_embeddings=True
        )

        # Mean pooling
        return np.mean(chunk_embeddings, axis=0)

    def embed(self, docs):
        embeddings = []
        for doc in docs:
            if not isinstance(doc, str) or len(doc.strip()) == 0:
                embeddings.append(
                    np.zeros(self.model.get_sentence_embedding_dimension())
                )
                continue

            emb = self._embed_chunks(doc)
            embeddings.append(emb)

        return np.vstack(embeddings)

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
        # Build DataFrame
        df = pd.DataFrame({"doc": docs, "topic": labels})
        df = df[df.topic != -1].reset_index(drop=True)

        if df.empty:
            return {}

        # Fit TF-IDF on all valid docs
        X = self.vectorizer.fit_transform(df["doc"])
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        topic_terms = {}

        # Compute top TF-IDF terms per topic
        for topic_id, group in df.groupby("topic"):
            idx = group.index.to_numpy()

            # Mean TF-IDF score for this topic
            tfidf_scores = X[idx].mean(axis=0).A1

            # Top-N terms
            top_idx = np.argsort(tfidf_scores)[::-1][: self.n_top_terms]
            keywords = feature_names[top_idx].tolist()

            topic_terms[topic_id] = keywords

        self.topic_terms_ = topic_terms
        return topic_terms

    def score_sentiment(self, docs):
        return [self.sia.polarity_scores(d)["compound"] for d in docs]

    def aggregate_sentiment(self, labels, sentiment):
        df = pd.DataFrame({"topic": labels, "sentiment": sentiment})
        df = df[df.topic != -1]
        return df.groupby("topic")["sentiment"].mean().to_dict()

    def _compute_noise_percentage(self, labels):
        labels = np.array(labels)
        total = len(labels)
        if total == 0:
            return 0.0
        noise_count = np.sum(labels == -1)
        return noise_count / total

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
        docs = pd.Series(self._extract_docs(data, column), dtype="string")

        docs = (
            docs.str.normalize("NFKC")  # normalize unicode
            .str.replace(r"\s+", " ", regex=True)  # collapse whitespace
            .str.strip()  # trim edges
        )

        embeddings = self.embed(docs)
        reduced = self.reduce(embeddings)
        raw_labels = self.cluster(reduced)

        # Save raw labels BEFORE reassignment
        df_raw = pd.DataFrame({"text": docs, "topic": raw_labels})

        noise_pct = self._compute_noise_percentage(raw_labels)
        print(f"Noise percentage: {noise_pct: .2f}")

        # Optional noise reassignment BEFORE renumbering
        if reassign_noise:
            df_temp = pd.DataFrame({"topic": raw_labels})
            reassigner = NoiseReassigner(df_temp, embeddings)
            df_temp = reassigner.reassign()
            raw_labels = df_temp["topic"].values

        # Renumber after reassignment
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

        # Word count
        df["word_count"] = (
            df["text"]
            .fillna("")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.split()
            .str.len()
        )

        # Token count
        df["token_count"] = df["text"].apply(
            lambda t: len(self.tokenizer.encode(t, add_special_tokens=True))
        )

        df["topic_keywords"] = df["topic"].apply(lambda t: topic_terms.get(t, []))
        df["topic_name"] = df["topic_keywords"].apply(
            lambda kws: ", ".join(kws[:3]) if kws else "Noise / Unassigned"
        )

        total_clusters: int = df["topic_name"].nunique()
        print(f"Total clusters: {total_clusters}")

        validator_final = Validator(df, embeddings)
        topic_coh, mean_coh = validator_final.topic_coherence()
        print(f"Mean coherence: {mean_coh:.3f}")

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
        for _, emb in enumerate(noise_embeddings):
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

    def export_to_excel(
        self, reps, output_path="outputs/representative_responses.xlsx"
    ):
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


def main() -> None:
    start_time: float = time.perf_counter()

    # Load and shape data
    df = load_df(
        DATA_CONFIG.filename,
        usecols=[DATA_CONFIG.usecol],
    )
    df = df.dropna().drop_duplicates()
    df = df.rename(columns={DATA_CONFIG.usecol: "text"})

    # 1. Run pipeline
    pipeline = TopicModelingPipeline(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        cluster_method="hdbscan",
        min_cluster_size=4,  # Lowering the value lowers noise but also decreseas coherence
        cluster_selection_method="leaf",
        n_components=10,
        n_neighbors=20,  # Lowering the value slightly increases noise but slightly increases coherence
    )

    print("Fitting data...")
    pipeline.fit(df, column="text")
    print("Transforming data...")
    df_out = pipeline.transform(df, column="text", reassign_noise=True)

    # 2. Compute embeddings
    print("Computing embeddings...")
    embeddings = pipeline.embed(df_out["text"].tolist())

    # 3. Noise reassignment
    reassigner = NoiseReassigner(df_out, embeddings)
    print("Reassigning noise...")
    df_out = reassigner.reassign()

    # 4. Export corrected topic assignments to Excel
    df_out.to_excel("outputs/topic_analysis.xlsx", index=False)

    # 5. Extract representative responses
    extractor = RepresentativeResponseExtractor(df_out, embeddings)
    print("Getting representative responses...")
    reps = extractor.get_representative_responses(top_k=3)
    extractor.export_to_excel(reps, "outputs/representative_responses.xlsx")

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
