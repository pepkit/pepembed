# %%
import os
import numpy as np
from qdrant_client import QdrantClient

# %%
# get the qdrant connection info
QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_PORT = os.environ.get("QDRANT_PORT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# connect to qdrant
qdrant = QdrantClient(
    url=QDRANT_HOST, 
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
    timeout=1000
)

# %%
# get number of embeddings
n_embeddings = qdrant.get_collection(collection_name="pephub").points_count

# %%
SAMPLE_SIZE = 10000
BATCH_SIZE = 10
# randomly sample embeddings using the qdrant scroll API
# in batches of 10
# generate a random offset
embeddings = []
for i in range(SAMPLE_SIZE // BATCH_SIZE):
    print(f"Batch {i}")
    offset = np.random.randint(0, n_embeddings - 10)
    result = qdrant.scroll(
        collection_name="pephub",
        limit=BATCH_SIZE,
        with_payload=False,
        with_vectors=True,
        offset=offset
    )
    embeddings.append(list(result)[0])

# %%
# flatten the list
embeddings = [e for batch in embeddings for e in batch]

# %%
embeddings = [np.array(e.vector) for e in embeddings]

# %%
from umap import UMAP

reducer = UMAP(n_components=2, random_state=42)
umap_embedding = reducer.fit_transform(embeddings)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

_, ax = plt.subplots(figsize=(5, 5))

plt.rcParams['figure.dpi'] = 300

sns.scatterplot(
    x=umap_embedding[:,0],
    y=umap_embedding[:,1],
    s=5,
    linewidth=0,
    ax=ax
)

ax.set_title("UMAP of GEO Sample Descriptions")
ax.set_xlabel("UMAP 1", fontsize=14)
ax.set_ylabel("UMAP 2", fontsize=14)

# %%
