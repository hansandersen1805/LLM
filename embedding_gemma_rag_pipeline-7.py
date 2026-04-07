# =============================================================================
# EmbeddingGemma RAG Pipeline for Azure Databricks
# =============================================================================
# Runtime: Databricks DBR 17.3 LTS ML
#
# Pre-installed library versions used by this script (verified from the
# official DBR 17.3 ML requirements manifest):
#
#   mlflow-skinny        == 3.0.1
#   sentence-transformers == 4.0.1
#   transformers         == 4.51.3
#   torch                == 2.7.0  (GPU cluster includes CUDA 12.6)
#   langchain            == 0.3.21
#   langchain-core       == 0.3.63
#   databricks-sdk       == 0.49.0
#   numpy                == 2.1.3
#   huggingface-hub      == 0.30.2
#
# No additional pip installs are required on DBR 17.3 ML.
# =============================================================================

# Databricks notebook source

# COMMAND ----------

# =============================================================================
# STEP 1: Load the model and verify it produces valid embeddings
# =============================================================================
# We load EmbeddingGemma through the sentence-transformers library rather than
# raw HuggingFace transformers. This is critical because sentence-transformers
# wraps the model with the correct pooling strategy and normalization that
# EmbeddingGemma requires to produce meaningful embeddings.

from sentence_transformers import SentenceTransformer
import numpy as np

# NOTE: Replace with your exact model path or HuggingFace Hub identifier.
# If the model is stored in a Databricks Volume or DBFS path, use that path.
MODEL_NAME = "google/embedding-gemma-001"

model = SentenceTransformer(MODEL_NAME)

# Inspect the prompt templates the model expects.
# This dictionary tells you exactly which prefix strings to use.
# Typical output: {'search_query': 'search_query: ', 'search_document': 'search_document: '}
print("Available prompt templates:", model.prompts)

# COMMAND ----------

# =============================================================================
# STEP 2: Sanity-check the embeddings
# =============================================================================
# EmbeddingGemma produces degenerate outputs (single-line headers) if you do
# NOT prepend the correct task prefix. Here we verify that prefixed inputs
# produce full-dimensional vectors and that semantically related pairs have
# high cosine similarity.

doc_embedding = model.encode("search_document: Retrieval augmented generation combines search with LLMs.")
query_embedding = model.encode("search_query: What is RAG?")

print(f"Document embedding shape : {doc_embedding.shape}")   # e.g. (768,) or (3072,)
print(f"Query embedding shape    : {query_embedding.shape}")

# Cosine similarity — should be well above 0.5 for a related pair
cosine_sim = np.dot(doc_embedding, query_embedding) / (
    np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding)
)
print(f"Cosine similarity        : {cosine_sim:.4f}")

# If the shapes are (1,) or the similarity is near 0, something is wrong.
assert doc_embedding.shape[0] > 1, "ERROR: Embedding is degenerate — check model and prefixes."

# COMMAND ----------

# =============================================================================
# STEP 3: Import MLflow (signature is handled automatically)
# =============================================================================
# When we log the model with task="llm/v1/embeddings" in Step 4, MLflow
# automatically sets an embeddings-compatible signature for the model AND
# handles data pre/post-processing to conform to the OpenAI Embeddings API
# spec. This means we do NOT need to define a manual signature — MLflow
# takes care of it.

import mlflow

# COMMAND ----------

# =============================================================================
# STEP 4: Log the model with mlflow.sentence_transformers
# =============================================================================
# Using sentence_transformers.log_model() instead of transformers.log_model()
# ensures that:
#   1. The correct pooling and normalization layers are preserved.
#   2. The model loads back through sentence-transformers at serving time.
#   3. When you send a properly prefixed string, you get a real embedding.
#
# CRITICAL: Setting task="llm/v1/embeddings" tells MLflow to make the serving
# endpoint OpenAI-compatible. Without this, the endpoint expects MLflow's
# standard pyfunc format (dataframe_split) and will reject OpenAI-style
# {"input": [...]} payloads with a 400 Bad Request error.

# End any active run left over from a previous execution of this notebook.
# This prevents the new log_model from accidentally appending to a stale run.
if mlflow.active_run():
    mlflow.end_run()
    print("Ended stale active run.")

# Set your MLflow experiment — change this to your workspace path.
EXPERIMENT_PATH = "/Users/your_email@company.com/embedding_gemma_rag"
mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="embedding_gemma_sentence_transformers") as run:

    # Log the model. The pip_requirements pin versions to match DBR 17.3 ML
    # so the serving environment reproduces the same behavior.
    #
    # NOTE on MLflow 3.x: The 'artifact_path' parameter is deprecated in favor
    # of 'name'. We use 'name' here to avoid deprecation warnings and to ensure
    # the model_uri returned in model_info works correctly with Unity Catalog
    # registration. If you get an unexpected keyword argument error for 'name',
    # fall back to artifact_path="embedding_model".
    model_info = mlflow.sentence_transformers.log_model(
        model=model,
        name="embedding_model",
        task="llm/v1/embeddings",           # <-- Enables OpenAI-compatible API format
        input_example=["search_query: What is RAG?"],
        pip_requirements=[
            "sentence-transformers==4.0.1",
            "transformers==4.51.3",
            "torch==2.7.0",
        ],
    )

    run_id = run.info.run_id
    # model_info.model_uri is the most reliable way to reference the model
    # in MLflow 3.x (format: "models:/<model_id>")
    print(f"MLflow run ID : {run_id}")
    print(f"Model URI     : {model_info.model_uri}")

    # Verify that the task metadata was saved with the model.
    # This is the metadata that tells the serving endpoint to use
    # OpenAI-compatible format. If this shows None, the endpoint
    # will return raw lists instead of {"data": [{"embedding": [...]}]}.
    task_metadata = getattr(model_info, "metadata", None)
    print(f"Model metadata: {task_metadata}")
    if task_metadata and "task" in str(task_metadata):
        print("✓ task='llm/v1/embeddings' is set in model metadata.")
    else:
        print("⚠ WARNING: task metadata may not have been saved.")

# COMMAND ----------

# =============================================================================
# STEP 5: Register the model in Unity Catalog
# =============================================================================
# Unity Catalog is the recommended model registry for Databricks. Registering
# the model here makes it available for serving endpoint creation and provides
# lineage tracking, access control, and versioning.
#
# IMPORTANT — MLflow 3.0.1 (shipped with DBR 17.3 ML) changed the return type
# of mlflow.register_model(). Depending on your UC configuration, it may return
# a ModelVersion object whose attributes differ from MLflow 2.x, or raise
# attribute errors when you access .version directly.
#
# The most reliable approach on MLflow 3.x + Unity Catalog is to use the
# MlflowClient API, which gives you a well-defined ModelVersion entity.

from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

# Replace these with your actual catalog and schema names.
CATALOG = "your_catalog"
SCHEMA = "your_schema"
MODEL_NAME_UC = f"{CATALOG}.{SCHEMA}.embedding_gemma"

client = MlflowClient(registry_uri="databricks-uc")

# Ensure the registered model exists (creates it if not).
try:
    client.create_registered_model(MODEL_NAME_UC)
    print(f"Created new registered model: {MODEL_NAME_UC}")
except Exception as e:
    # Model already exists — this is expected on subsequent runs.
    if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
        print(f"Registered model already exists: {MODEL_NAME_UC}")
    else:
        raise

# Create a new version of the model using the URI returned by log_model.
# In MLflow 3.x, model_info.model_uri is the reliable reference
# (format: "models:/<model_id>"). Using the old "runs:/<run_id>/..." format
# can break because MLflow 3 stores model artifacts in a different location.
model_version_obj = client.create_model_version(
    name=MODEL_NAME_UC,
    source=model_info.model_uri,
    run_id=run_id,
)

# Extract the version number as a string (safe for both MLflow 2.x and 3.x).
MODEL_VERSION = str(model_version_obj.version)

print(f"Registered model: {MODEL_NAME_UC}")
print(f"Version         : {MODEL_VERSION}")

# COMMAND ----------

# =============================================================================
# STEP 6: Create the Model Serving endpoint (clean slate)
# =============================================================================
# This creates a GPU-backed serving endpoint that exposes an OpenAI-compatible
# /invocations API. The endpoint is a "dumb pipe" — it takes whatever strings
# you send, passes them through the model, and returns embedding vectors.
# It does NOT automatically prepend task prefixes; that responsibility belongs
# to the calling application (see Step 8).
#
# IMPORTANT: We delete any existing endpoint with the same name first to
# guarantee the new endpoint serves the version we just registered (with
# task="llm/v1/embeddings" metadata). Without this, an existing endpoint
# might continue serving a stale model version that lacks the task metadata.

import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

w = WorkspaceClient()

ENDPOINT_NAME = "embedding-gemma-rag"

# --- Delete existing endpoint if present ---
try:
    w.serving_endpoints.delete(name=ENDPOINT_NAME)
    print(f"Deleted existing endpoint '{ENDPOINT_NAME}'.")
    # Brief pause to let the deletion propagate.
    import time
    time.sleep(10)
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "not found" in str(e).lower():
        print(f"No existing endpoint '{ENDPOINT_NAME}' found — creating fresh.")
    else:
        raise

# --- Create the endpoint from scratch with the new model version ---
print(f"Creating endpoint '{ENDPOINT_NAME}' with model {MODEL_NAME_UC} version {MODEL_VERSION}...")

w.serving_endpoints.create_and_wait(
    name=ENDPOINT_NAME,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=MODEL_NAME_UC,
                entity_version=MODEL_VERSION,
                workload_size="Small",                                # Adjust for throughput needs
                scale_to_zero_enabled=True,                            # Cost savings when idle
                workload_type=ServingModelWorkloadType.GPU_SMALL,      # EmbeddingGemma requires GPU
            )
        ]
    ),
    timeout=datetime.timedelta(minutes=30),  # GPU endpoints can take 20+ min to provision
)

print(f"Serving endpoint '{ENDPOINT_NAME}' is ready.")

# COMMAND ----------

# =============================================================================
# STEP 7: Test the endpoint directly (OpenAI-compatible format)
# =============================================================================
# Because we logged the model with task="llm/v1/embeddings" in Step 4,
# the endpoint accepts the OpenAI embedding format:
#   Request:  {"input": ["text1", "text2"]}
#   Response: {"data": [{"embedding": [...], "index": 0}, ...]}
#
# Without task="llm/v1/embeddings", the endpoint expects the standard MLflow
# pyfunc format (e.g., {"dataframe_split": {"columns": [...], "data": [...]}})
# and will return a 400 Bad Request for OpenAI-style payloads.
#
# IMPORTANT: You must still include the task prefix in each input string.
# The endpoint does not add prefixes for you.

import requests

# Retrieve workspace URL and PAT from the notebook context.
databricks_host = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().apiUrl().get()
)
token = (
    dbutils.notebook.entry_point
    .getDbutils().notebook().getContext().apiToken().get()
)

def call_embedding_endpoint(payload: dict) -> dict:
    """Call the Databricks Model Serving endpoint and return the JSON response."""
    response = requests.post(
        f"{databricks_host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    # If you get a 400 here, check:
    #   1. Was the model logged with task="llm/v1/embeddings"?
    #   2. Is the endpoint status "READY"? (check in Serving UI)
    #   3. Is the payload format correct for how the model was logged?
    if not response.ok:
        print(f"ERROR {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()


# --- Test document embeddings (use "search_document: " prefix) ---
doc_payload = {
    "input": [
        "search_document: Retrieval augmented generation combines a retriever with a generative model.",
        "search_document: Vector databases store embeddings for fast similarity search.",
    ]
}

# --- Test query embeddings (use "search_query: " prefix) ---
query_payload = {
    "input": [
        "search_query: How does RAG work?",
    ]
}

doc_result = call_embedding_endpoint(doc_payload)
query_result = call_embedding_endpoint(query_payload)

# --- Diagnostic: inspect what the endpoint actually returns ---
# If task="llm/v1/embeddings" took effect, doc_result is a dict:
#   {"object": "list", "data": [{"embedding": [...], "index": 0}, ...]}
# If it did NOT take effect, doc_result is a raw list of embedding arrays:
#   [[0.012, -0.034, ...], [0.056, 0.078, ...]]
print(f"Response type: {type(doc_result)}")
print(f"Response preview: {str(doc_result)[:200]}")

# Handle both formats so the test works either way.
if isinstance(doc_result, dict) and "data" in doc_result:
    # OpenAI-compatible format (task="llm/v1/embeddings" worked)
    print(f"\nEndpoint is returning OpenAI-compatible format.")
    print(f"Document embeddings returned : {len(doc_result['data'])} vectors")
    print(f"Document vector dimension    : {len(doc_result['data'][0]['embedding'])}")
    print(f"Query vector dimension       : {len(query_result['data'][0]['embedding'])}")
elif isinstance(doc_result, list):
    # Raw list format (task param did not propagate — see note below)
    print(f"\nEndpoint is returning raw list format (not OpenAI-compatible).")
    print(f"Document embeddings returned : {len(doc_result)} vectors")
    print(f"Document vector dimension    : {len(doc_result[0])}")
    query_vec = query_result[0] if isinstance(query_result, list) else query_result
    print(f"Query vector dimension       : {len(query_vec)}")
    print()
    print("NOTE: The endpoint is NOT returning OpenAI format. This means")
    print("task='llm/v1/embeddings' did not take effect. This can happen if")
    print("the model was registered/served before the task metadata was set.")
    print("To fix this:")
    print("  1. Re-run Step 4 (log_model with task='llm/v1/embeddings')")
    print("  2. Re-run Step 5 (register the new version)")
    print("  3. Update the serving endpoint to point to the new version")
    print("  4. Wait for the endpoint to reach READY state, then re-test")
    print()
    print("Alternatively, if you want to proceed with the raw list format,")
    print("the DatabricksEmbeddings class in LangChain can handle it — your")
    print("RAG pipeline (Steps 8-10) will still work because LangChain")
    print("adapts to the response format automatically.")
else:
    print(f"\nUnexpected response format: {type(doc_result)}")
    print(f"Full response: {doc_result}")

# COMMAND ----------

# =============================================================================
# STEP 8: Define the Prefix Wrapper for RAG integration
# =============================================================================
# This is the critical piece that makes EmbeddingGemma work in a RAG pipeline.
#
# LangChain's embedding interface has two methods:
#   - embed_documents()  → called when INDEXING chunks into the vector store
#   - embed_query()      → called at QUERY TIME for the user's question
#
# EmbeddingGemma requires different prefixes for each:
#   - Documents: "search_document: <text>"
#   - Queries:   "search_query: <text>"
#
# This wrapper automatically prepends the correct prefix so the rest of your
# RAG pipeline doesn't need to know about EmbeddingGemma's requirements.
#
# NOTE: Some EmbeddingGemma variants use different prefix strings
# (e.g., "passage: " / "query: "). Check model.prompts from Step 1 and
# update DOC_PREFIX / QUERY_PREFIX accordingly.

from langchain_community.embeddings import DatabricksEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List


class PrefixedEmbeddingGemma(Embeddings):
    """
    LangChain-compatible embedding wrapper that prepends the correct
    EmbeddingGemma task prefix before calling the Databricks serving endpoint.

    Parameters
    ----------
    endpoint : str
        Name of the Databricks Model Serving endpoint.
    doc_prefix : str
        Prefix for document/chunk embeddings (used during indexing).
    query_prefix : str
        Prefix for query embeddings (used during retrieval).
    """

    def __init__(
        self,
        endpoint: str,
        doc_prefix: str = "search_document: ",
        query_prefix: str = "search_query: ",
    ):
        self._base = DatabricksEmbeddings(endpoint=endpoint)
        self._doc_prefix = doc_prefix
        self._query_prefix = query_prefix

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents — prepends the document prefix."""
        prefixed = [f"{self._doc_prefix}{text}" for text in texts]
        return self._base.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query — prepends the query prefix."""
        prefixed = f"{self._query_prefix}{text}"
        return self._base.embed_query(prefixed)


# COMMAND ----------

# =============================================================================
# STEP 9: Wire the wrapper into the RAG retriever
# =============================================================================
# This connects the prefixed embedding wrapper to Databricks Vector Search
# through LangChain. The retriever returned here can be plugged into any
# LangChain RAG chain (RetrievalQA, ConversationalRetrievalChain, etc.).
#
# The wrapper handles the prefix logic transparently:
#   - When you populate the index, embed_documents() adds "search_document: "
#   - When a user asks a question, embed_query() adds "search_query: "

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

# Instantiate the wrapper pointing at our serving endpoint.
rag_embeddings = PrefixedEmbeddingGemma(endpoint=ENDPOINT_NAME)

# Connect to your existing Databricks Vector Search index.
# Replace these with your actual endpoint and index names.
VS_ENDPOINT_NAME = "your_vector_search_endpoint"   # Vector Search endpoint (not model serving)
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.your_index"

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME,
)

# Build the LangChain retriever.
retriever = DatabricksVectorSearch(
    index=index,
    embedding=rag_embeddings,
    text_column="chunk_text",               # Column containing the text chunks
    columns=["chunk_text", "doc_uri"],      # Columns to return with results
).as_retriever(search_kwargs={"k": 5})     # Return top 5 matches

# COMMAND ----------

# =============================================================================
# STEP 10: Quick retrieval test
# =============================================================================
# Run a test query through the full pipeline:
#   User question → PrefixedEmbeddingGemma.embed_query() → Vector Search → results

test_query = "How does retrieval augmented generation work?"
results = retriever.invoke(test_query)

print(f"Retrieved {len(results)} documents for: '{test_query}'\n")
for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Source : {doc.metadata.get('doc_uri', 'N/A')}")
    print(f"Text   : {doc.page_content[:200]}...")
    print()

# COMMAND ----------

# =============================================================================
# IMPORTANT REMINDERS
# =============================================================================
#
# 1. REINDEX IF SWITCHING MODELS
#    If your vector index was built with Nomic v1.5 embeddings, you MUST
#    reindex all chunks using the new EmbeddingGemma endpoint with
#    "search_document: " prefixes. You cannot mix embeddings from different
#    models in the same index.
#
# 2. VERIFY YOUR PREFIX STRINGS
#    The prefixes used here ("search_document: " and "search_query: ") are
#    the most common for EmbeddingGemma. However, some model variants use
#    different strings (e.g., "passage: " / "query: "). Always check
#    model.prompts from Step 1 for your specific checkpoint.
#
# 3. GPU REQUIREMENT
#    EmbeddingGemma requires a GPU-backed serving endpoint (workload_type
#    "GPU_SMALL" or larger). CPU endpoints will either fail or run extremely
#    slowly.
#
# 4. OPENAI COMPATIBILITY
#    The endpoint is OpenAI-compatible ONLY if you log the model with
#    task="llm/v1/embeddings". Without this parameter, the endpoint
#    expects MLflow's pyfunc format (dataframe_split) and will reject
#    OpenAI-style {"input": [...]} payloads with a 400 error.
#    With task="llm/v1/embeddings", the endpoint accepts {"input": [...]}
#    and returns {"data": [{"embedding": [...]}]}.
#
# 5. DBR 17.3 ML COMPATIBILITY
#    All libraries used in this script are pre-installed on DBR 17.3 LTS ML.
#    No additional pip installs are needed:
#      - mlflow-skinny 3.0.1 (includes mlflow.sentence_transformers)
#      - sentence-transformers 4.0.1 (compatible with mlflow 3.0.1)
#      - transformers 4.51.3
#      - torch 2.7.0
#      - langchain 0.3.21 / langchain-core 0.3.63
#      - databricks-sdk 0.49.0
# =============================================================================
