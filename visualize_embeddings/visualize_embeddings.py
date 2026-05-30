import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import hdbscan

X = np.load("visualize_embeddings/astronomy_embeddings.npy")
df = pd.read_pickle("visualize_embeddings/astronomy_chunks.pkl")
texts = df["text"].tolist()

X_pca = PCA(n_components=50).fit_transform(X)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5
)
clusters = clusterer.fit_predict(X_pca)

X_tsne = TSNE(
    n_components=2,
    random_state=42
).fit_transform(X_pca)

plt.figure(figsize=(10, 8))
for cluster_id in np.unique(clusters):
    mask = clusters == cluster_id
    label = (
        "šum" if cluster_id == -1
        else f"grupa {cluster_id}"
    )
    plt.scatter(
        X_tsne[mask, 0], X_tsne[mask, 1],
        label=label, s=30
    )

plt.title("t-SNE s HDBSCAN grupiranjem")
plt.legend(
    title="Grupe",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.xlabel("t-SNE 1. komponenta")
plt.ylabel("t-SNE 2. komponenta")
plt.tight_layout()
plt.show()