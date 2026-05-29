## Topic Modeling & Sentiment Analysis Pipeline

A embedding‑based topic modeling system with UMAP reduction, HDBSCAN clustering, noise reassignment, sentiment scoring, and validation.

## Overview

This tool provides topic‑modeling pipeline designed for short-form text, such as survey or request for information responses. It uses:
- Sentence embeddings for semantic representation
- UMAP for dimensionality reduction
- HDBSCAN for density‑based clustering
- Noise reassignment to recover meaningful topics
- Sentiment analysis for emotional context
- A dedicated Validation module for model quality assessment

The pipeline produces:
- Topic assignments
- Topic keywords
- Topic names
- Sentiment scores
- Word/token counts
- Full validation metrics
 
## Pipeline Architecture 

1. Preprocessing: Text normalization, chunking for long responses, etc.
2. Embedding: Users sentence-transformer models to develop embeddings
3. Dimensionality Reduction: Uses UMAP for dimensionality reduction.
4. Clustering: Uses HDBSCAN to identify clusters in UMAP space.
5. Noise Reassignment: Embeddings labeled as noise reassigned to nearest cluster centroid using Kmeans.
6. Topic Keyword Extraction: Keywords are aggregated per topic using TF-IDF.
7. Sentiment Analysis: Generates sentiment analysis using NLTK Vader.
8. Validation: Computes noise percentage, topic coherence, and cluster size distribution

## Model



## Chuck-Aware Embedding


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

Requires the following properly-formatted files to run:
```
pd3po.xlsx
nsf_award_search.xlsx
baa.xlsx
team_data.xlsx
CARegionMappingData.xlsx
AUTH.py
```

Install the required libraries:

	pip install -r requirements.txt

Contact codeowners to obtain required files/formats.