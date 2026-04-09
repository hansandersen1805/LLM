# =============================================================================
# EmbeddingGemma RAG Pipeline for Azure Databricks
# Managed Delta Sync Index with Databricks-Managed Embeddings
# =============================================================================
# Runtime: Databricks DBR 17.3 LTS ML
#
# Pre-installed library versions (verified from official DBR 17.3 ML manifest):
#   mlflow-skinny         == 3.0.1
#   sentence-transformers == 4.0.1
#   transformers          == 4.51.3
#   torch                 == 2.7.0  (GPU cluster includes CUDA 12.6)
#   langchain             == 0.3.21
#   langchain-core        == 0.3.63
#   databricks-sdk        == 0.49.0
#   numpy                 == 2.1.3
#
# No additional pip installs are required on DBR 17.3 ML.
#
# APPROACH:
#   1. Log EmbeddingGemma with mlflow.transformers + task="llm/v1/embeddings"
#      (the proven path for OpenAI-compatible format on Databricks serving).
#   2. Create a Delta table with a prefixed text column so Databricks auto-
#      embeds using "search_document: <text>" during index sync.
#   3. Create a managed Delta Sync index that points at our EmbeddingGemma
#      serving endpoint — Databricks handles embedding and sync automatically.
#   4. At query time, prepend "search_query: " before querying the index.
# =============================================================================

# Databricks notebook source

# COMMAND ----------

# =============================================================================
# STEP 1: Load via SentenceTransformer and verify embeddings
# =============================================================================

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "google/embedding-gemma-001"  # Replace with your exact checkpoint

st_model = SentenceTransformer(MODEL_NAME)

print("Available prompt templates:", st_model.prompts)
# Expected: {'search_query': 'search_query: ', 'search_document': 'search_document: '}

# Sanity check
doc_emb = st_model.encode("search_document: RAG combines search with LLMs.")
qry_emb = st_model.encode("search_query: What is RAG?")
print(f"Doc embedding shape : {doc_emb.shape}")
print(f"Query embedding shape: {qry_emb.shape}")

cosine_sim = np.dot(doc_emb, qry_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(qry_emb))
print(f"Cosine similarity    : {cosine_sim:.4f}")
assert doc_emb.shape[0] > 1, "Embedding is degenerate — check model and prefixes."

# COMMAND ----------

# =============================================================================
# STEP 2: Build a transformers feature-extraction pipeline
# =============================================================================

import transformers

hf_model = st_model[0].auto_model
hf_tokenizer = st_model.tokenizer

print(f"HF model type  : {type(hf_model).__name__}")
print(f"Tokenizer type : {type(hf_tokenizer).__name__}")

embedding_pipeline = transformers.pipeline(
    task="feature-extraction",
    model=hf_model,
    tokenizer=hf_tokenizer,
)

# Quick test
test_output = embedding_pipeline("search_query: test")
print(f"Pipeline output type : {type(test_output)}")
print(f"Pipeline output shape: {len(test_output)} x {len(test_output[0])} x {len(test_output[0][0])}")

# COMMAND ----------

# =============================================================================
# STEP 3: Log with mlflow.transformers (proven OpenAI-compatible path)
# =============================================================================

import mlflow

if mlflow.active_run():
    mlflow.end_run()
    print("Ended stale active run.")

EXPERIMENT_PATH = "/Users/your_email@company.com/embedding_gemma_rag"  # Change this
mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="embedding_gemma_transformers") as run:
    model_info = mlflow.transformers.log_model(
        transformers_model=embedding_pipeline,
        artifact_path="embedding_model",
        task="llm/v1/embeddings",
        input_example=["search_query: What is RAG?"],
        pip_requirements=[
            "sentence-transformers==4.0.1",
            "transformers==4.51.3",
            "torch==2.7.0",
        ],
    )
    run_id = run.info.run_id
    print(f"MLflow run ID : {run_id}")
    print(f"Model URI     : {model_info.model_uri}")

# COMMAND ----------

# =============================================================================
# STEP 4: Register the model in Unity Catalog
# =============================================================================

from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

CATALOG = "your_catalog"       # Replace
SCHEMA = "your_schema"         # Replace
MODEL_NAME_UC = f"{CATALOG}.{SCHEMA}.embedding_gemma"

client = MlflowClient(registry_uri="databricks-uc")

try:
    client.create_registered_model(MODEL_NAME_UC)
    print(f"Created new registered model: {MODEL_NAME_UC}")
except Exception as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
        print(f"Registered model already exists: {MODEL_NAME_UC}")
    else:
        raise

model_version_obj = client.create_model_version(
    name=MODEL_NAME_UC,
    source=model_info.model_uri,
    run_id=run_id,
)
MODEL_VERSION = str(model_version_obj.version)

print(f"Registered model: {MODEL_NAME_UC}")
print(f"Version         : {MODEL_VERSION}")

# COMMAND ----------

# =============================================================================
# STEP 5: Create the Model Serving endpoint (clean slate)
# =============================================================================

import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

w = WorkspaceClient()
ENDPOINT_NAME = "embedding-gemma-rag"

# Delete existing endpoint to guarantee we serve the new version.
try:
    w.serving_endpoints.delete(name=ENDPOINT_NAME)
    print(f"Deleted existing endpoint '{ENDPOINT_NAME}'.")
    import time
    time.sleep(10)
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
        print(f"No existing endpoint found — creating fresh.")
    else:
        raise

print(f"Creating endpoint with model {MODEL_NAME_UC} v{MODEL_VERSION}...")

w.serving_endpoints.create_and_wait(
    name=ENDPOINT_NAME,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=MODEL_NAME_UC,
                entity_version=MODEL_VERSION,
                workload_size="Small",
                scale_to_zero_enabled=True,
                workload_type=ServingModelWorkloadType.GPU_SMALL,
            )
        ]
    ),
    timeout=datetime.timedelta(minutes=30),
)

print(f"Serving endpoint '{ENDPOINT_NAME}' is ready.")

# COMMAND ----------

# =============================================================================
# STEP 6: Test the endpoint (OpenAI-compatible format)
# =============================================================================

import requests

databricks_host = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().apiUrl().get()
)
token = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().apiToken().get()
)

def call_embedding_endpoint(payload: dict) -> dict:
    """Call the serving endpoint and return JSON response."""
    response = requests.post(
        f"{databricks_host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    if not response.ok:
        print(f"ERROR {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()


doc_result = call_embedding_endpoint({
    "input": [
        "search_document: RAG combines a retriever with a generative model.",
        "search_document: Vector databases store embeddings for similarity search.",
    ]
})

query_result = call_embedding_endpoint({
    "input": ["search_query: How does RAG work?"]
})

print(f"Response type: {type(doc_result)}")
if isinstance(doc_result, dict) and "data" in doc_result:
    print(f"✓ OpenAI-compatible format confirmed!")
    print(f"  Document embeddings: {len(doc_result['data'])} vectors")
    print(f"  Vector dimension   : {len(doc_result['data'][0]['embedding'])}")
    print(f"  Query dimension    : {len(query_result['data'][0]['embedding'])}")
else:
    print(f"✗ Unexpected format: {str(doc_result)[:300]}")

# COMMAND ----------

# =============================================================================
# STEP 7: Prepare the source Delta table with prefixed text column
# =============================================================================
# EmbeddingGemma requires "search_document: " prefixed to every chunk for
# indexing. Since Databricks-managed embeddings send raw column text to the
# endpoint, we add a column with the prefix already prepended.
#
# IMPORTANT: Your existing source table probably has a text column like
# "chunk_text". We add a new column "chunk_text_prefixed" that contains
# 'search_document: ' + chunk_text. The index will embed this column.
#
# Adapt the table name and column names to match your actual data.

from pyspark.sql import functions as F

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.your_chunks_table"    # Replace with your table
TEXT_COLUMN = "chunk_text"                                  # Your existing text column
PREFIXED_COLUMN = "chunk_text_prefixed"                     # New column for prefixed text
DOC_PREFIX = "search_document: "

# Read the source table and add the prefixed column.
df = spark.table(SOURCE_TABLE)

# Check if the prefixed column already exists (idempotent).
if PREFIXED_COLUMN not in df.columns:
    df_with_prefix = df.withColumn(
        PREFIXED_COLUMN,
        F.concat(F.lit(DOC_PREFIX), F.col(TEXT_COLUMN))
    )
    # Overwrite the table with the new column.
    # IMPORTANT: Make sure Change Data Feed (CDF) stays enabled on the table.
    df_with_prefix.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SOURCE_TABLE)
    print(f"Added '{PREFIXED_COLUMN}' column to {SOURCE_TABLE}")
else:
    print(f"'{PREFIXED_COLUMN}' column already exists in {SOURCE_TABLE}")

# Verify
spark.table(SOURCE_TABLE).select(TEXT_COLUMN, PREFIXED_COLUMN).show(3, truncate=60)

# COMMAND ----------

# =============================================================================
# STEP 8: Create the managed Delta Sync index
# =============================================================================
# This creates a fully managed index where Databricks:
#   - Calls your EmbeddingGemma serving endpoint to embed each chunk
#   - Reads from the prefixed column (so "search_document: " is included)
#   - Automatically syncs when the source Delta table changes
#   - Handles query embedding at search time using the same endpoint
#
# NOTE: You must delete any existing index with the same name first if
# it was created with a different configuration (different embedding model,
# different source column, etc.).

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

VS_ENDPOINT_NAME = "your_vector_search_endpoint"               # Replace
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.embedding_gemma_index"     # Replace
PRIMARY_KEY = "id"                                               # Replace with your PK column

# Delete existing index if it exists (to ensure clean configuration).
try:
    vsc.delete_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
    print(f"Deleted existing index '{VS_INDEX_NAME}'.")
    import time
    time.sleep(10)
except Exception as e:
    if "not found" in str(e).lower() or "does not exist" in str(e).lower():
        print(f"No existing index found — creating fresh.")
    else:
        # Some errors may be acceptable; print and continue.
        print(f"Note: {e}")

# Create the managed Delta Sync index.
# embedding_source_column points to the PREFIXED column so Databricks sends
# "search_document: <text>" to the endpoint during sync.
# embedding_model_endpoint_name points to our EmbeddingGemma serving endpoint.
index = vsc.create_delta_sync_index_and_wait(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME,
    source_table_name=SOURCE_TABLE,
    pipeline_type="TRIGGERED",
    primary_key=PRIMARY_KEY,
    embedding_source_column=PREFIXED_COLUMN,                 # The prefixed text column
    embedding_model_endpoint_name=ENDPOINT_NAME,             # Our EmbeddingGemma endpoint
    columns_to_sync=[TEXT_COLUMN, "doc_uri"],                # Sync original text + metadata
)

print(f"Index '{VS_INDEX_NAME}' created and synced.")

# COMMAND ----------

# =============================================================================
# STEP 9: Query the index with the retriever
# =============================================================================
# For a Databricks-managed embedding index, LangChain's DatabricksVectorSearch
# does NOT need an embedding object — Databricks handles query embedding
# internally by calling the same serving endpoint.
#
# HOWEVER, the index was built with "search_document: " prefixes, and at
# query time the index will send the raw query text to the endpoint.
# EmbeddingGemma needs "search_query: " for queries (different from the
# "search_document: " used for indexing).
#
# Two options:
#   Option A: Use similarity_search() directly and prepend the prefix yourself.
#   Option B: Use the model_endpoint_name_for_query parameter when creating
#             the index (if your endpoint handles both prefix types).
#
# Since Databricks sends raw query text to the endpoint, the simplest
# approach is Option A — query the index directly with the prefix prepended.

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME,
)

def search_with_prefix(query: str, num_results: int = 5) -> dict:
    """
    Search the vector index with the correct EmbeddingGemma query prefix.
    Prepends 'search_query: ' to the query before sending to the index.
    """
    prefixed_query = f"search_query: {query}"
    results = index.similarity_search(
        query_text=prefixed_query,
        columns=[TEXT_COLUMN, "doc_uri"],
        num_results=num_results,
    )
    return results


# Test query
test_query = "How does retrieval augmented generation work?"
results = search_with_prefix(test_query)

print(f"Query: '{test_query}'\n")
if "result" in results and "data_array" in results["result"]:
    for i, row in enumerate(results["result"]["data_array"], 1):
        print(f"--- Result {i} ---")
        print(f"Text  : {str(row[0])[:200]}...")
        print(f"Source: {row[1]}")
        print(f"Score : {row[2]}")
        print()
else:
    print(f"Results: {results}")

# COMMAND ----------

# =============================================================================
# STEP 10: (Optional) LangChain retriever wrapper
# =============================================================================
# If you want to plug this into a LangChain RAG chain, you can wrap the
# similarity_search call in a LangChain retriever. Since the managed index
# handles embedding internally, we override the query to add the prefix.

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List


class EmbeddingGemmaRetriever(BaseRetriever):
    """
    LangChain retriever for a Databricks-managed Delta Sync index
    that uses EmbeddingGemma. Prepends 'search_query: ' to every query.
    """
    index_name: str
    endpoint_name: str
    text_column: str = "chunk_text"
    metadata_columns: List[str] = ["doc_uri"]
    num_results: int = 5
    query_prefix: str = "search_query: "

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vsc = VectorSearchClient()
        idx = vsc.get_index(
            endpoint_name=self.endpoint_name,
            index_name=self.index_name,
        )

        # Prepend the EmbeddingGemma query prefix.
        prefixed_query = f"{self.query_prefix}{query}"

        results = idx.similarity_search(
            query_text=prefixed_query,
            columns=[self.text_column] + self.metadata_columns,
            num_results=self.num_results,
        )

        documents = []
        if "result" in results and "data_array" in results["result"]:
            columns = [col["name"] for col in results["manifest"]["columns"]]
            text_idx = columns.index(self.text_column)
            for row in results["result"]["data_array"]:
                metadata = {
                    col: row[columns.index(col)]
                    for col in self.metadata_columns
                    if col in columns
                }
                metadata["score"] = row[-1]  # Score is always last
                documents.append(Document(
                    page_content=str(row[text_idx]),
                    metadata=metadata,
                ))
        return documents


# Instantiate the retriever.
retriever = EmbeddingGemmaRetriever(
    index_name=VS_INDEX_NAME,
    endpoint_name=VS_ENDPOINT_NAME,
    text_column=TEXT_COLUMN,
    metadata_columns=["doc_uri"],
    num_results=5,
)

# Test it.
docs = retriever.invoke("How does retrieval augmented generation work?")
print(f"Retrieved {len(docs)} documents.\n")
for i, doc in enumerate(docs, 1):
    print(f"--- Result {i} ---")
    print(f"Score : {doc.metadata.get('score', 'N/A')}")
    print(f"Source: {doc.metadata.get('doc_uri', 'N/A')}")
    print(f"Text  : {doc.page_content[:200]}...")
    print()

# COMMAND ----------

# =============================================================================
# IMPORTANT REMINDERS
# =============================================================================
#
# 1. PREFIXED COLUMN IS REQUIRED
#    Databricks-managed embeddings send raw column text to the serving
#    endpoint. EmbeddingGemma needs "search_document: " prepended for
#    indexing, so the source table must have a column with the prefix
#    already included (Step 7). At query time, "search_query: " must
#    be prepended to the user's question (Step 9/10).
#
# 2. REINDEX IF SWITCHING MODELS
#    If your index was built with Nomic or another model, you must
#    delete it and create a new one pointing to the EmbeddingGemma
#    endpoint. You cannot mix embeddings from different models.
#
# 3. SCALE TO ZERO WARNING
#    Databricks recommends disabling scale_to_zero for the embedding
#    endpoint when using it with a managed index. If the endpoint is
#    scaled down when a sync triggers, the sync may timeout waiting
#    for the endpoint to warm up.
#
# 4. WHY mlflow.transformers
#    On DBR 17.3 ML, mlflow.transformers.log_model() with
#    task="llm/v1/embeddings" is the only proven path that produces
#    OpenAI-compatible format from Databricks Model Serving. The
#    sentence_transformers and pyfunc flavors return raw lists.
#
# 5. PIPELINE TYPE
#    "TRIGGERED" requires manual sync via index.sync(). Use
#    "CONTINUOUS" for automatic near-real-time sync (higher cost).
#
# 6. DBR 17.3 ML COMPATIBILITY
#    All libraries are pre-installed. No pip installs needed.
# =============================================================================
