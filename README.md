## Topic Modeling & Sentiment Analysis Pipeline

A embedding‑based topic modeling system with UMAP reduction, HDBSCAN clustering, noise reassignment, sentiment scoring, and validation.

## Overview

This tool provides topic‑modeling pipeline designed for short-form text, such as survey or request for information responses. It uses:
- Sentence embeddings for semantic representation
- UMAP for dimensionality reduction
- HDBSCAN for density‑based clustering
- Noise reassignment to recover meaningful topics
- Sentiment analysis for emotional context
- Validation analysis for model quality assessment

The pipeline produces:
- Topic assignments
- Topic keywords
- Topic names
- Sentiment scores
- Word/token counts
- Full validation metrics
- Vizualizations of model outputs
 
## Pipeline Architecture 

1. Preprocessing: Text normalization, chunking for long responses, etc.
2. Embedding: Users sentence-transformer models to develop embeddings
3. Dimensionality Reduction: Uses UMAP for dimensionality reduction.
4. Clustering: Uses HDBSCAN to identify clusters in UMAP space.
5. Noise Reassignment: Embeddings labeled as noise reassigned to Nearest-Centroid noise reassignment with cosine similarity threshold.
6. Topic Keyword Extraction: Keywords are aggregated per topic using c-TF-IDF and semantic re-ranking.
7. Sentiment Analysis: Generates sentiment analysis using NLTK Vader.
8. Validation: Computes noise percentage, mean topic coherence, cluster size distribution, reassignment percentage, and cluster confidence.
9. Vizualization: Produces vizualizations to assess topic density and distribution, topic similarity and keyword overlap, and umap embedding and confidence.

## Model

The pipeline is tuned for sentence-transformer/all-mpnet-base-v2, but can default to sentence-transformer/all-MiniLM-L6-v2 if no model parameter is provided. 

## Chuck-Aware Embedding

Since the token limit for mpnet models is limited to 256, the pipeline will automatically chuck documents longer than 350 and overlap the embeddings by 50 tokens to preserve semantic continutity. As such, text that exceeds the max context window can still be processed by the pipeline. If a non-npnet model is used, the `max_tokens` parameter may need to be adjusted.

## Parameters

The most important configuration parameters are:

`n_neighbors` (UMAP): Controls global vs local structure
- Recommended default is 20 as a starting point
- Higher: smoother space, fewer clusters, higher coherence
- Lower: more local structure, more clusters, lower coherence

`min_cluster_size` (HDBSCAN): Controls minimum topic size
- Recommended default is 4 as a starting point
- Higher: fewer, cleaner topics
- Lower: more topics, more noise

`cluster_selection_method` (HDBSCAN)
- Recommended default is "leaf"

## Requirements

Install the required libraries:

	pip install -r requirements.txt
