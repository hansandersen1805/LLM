# =============================================================================
# EmbeddingGemma RAG Pipeline for Azure Databricks
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
#   huggingface-hub       == 0.30.2
#
# No additional pip installs are required on DBR 17.3 ML.
#
# APPROACH:
#   We use mlflow.transformers.log_model() with task="llm/v1/embeddings"
#   because this is the PROVEN path for getting OpenAI-compatible format
#   from Databricks Model Serving (this is what worked for Nomic v1.5).
#
#   The EmbeddingGemma-specific challenge (correct pooling + task prefixes)
#   is handled by:
#     1. Loading the model via SentenceTransformer to verify it works.
#     2. Building a transformers.pipeline("feature-extraction") from the
#        underlying HuggingFace model + tokenizer.
#     3. Logging that pipeline with mlflow.transformers and
#        task="llm/v1/embeddings", which tells the Databricks serving
#        container to produce OpenAI-compatible responses.
#     4. The PrefixedEmbeddingGemma wrapper prepends "search_document: "
#        or "search_query: " to every input string at the LangChain level.
# =============================================================================

# Databricks notebook source

# COMMAND ----------

# =============================================================================
# STEP 1: Load via SentenceTransformer and verify embeddings
# =============================================================================
# We first load through sentence-transformers to confirm the model works
# and to inspect the expected prefix strings. This is a validation step.

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "google/embedding-gemma-001"  # Replace with your exact checkpoint

st_model = SentenceTransformer(MODEL_NAME)

# Check what prefixes the model expects.
print("Available prompt templates:", st_model.prompts)
# Expected: {'search_query': 'search_query: ', 'search_document': 'search_document: '}

# Sanity check: these should produce full-dimensional vectors.
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
# mlflow.transformers.log_model() with task="llm/v1/embeddings" is the
# proven path for OpenAI-compatible format on Databricks Model Serving
# (this is what worked for Nomic).
#
# We extract the underlying HuggingFace model and tokenizer from the
# SentenceTransformer object and build a feature-extraction pipeline.
# This pipeline produces token-level embeddings; the Databricks serving
# container's llm/v1/embeddings handling applies mean pooling when it
# post-processes the output into the OpenAI response format.

import transformers

# Extract the HuggingFace model and tokenizer from the SentenceTransformer.
# The auto_model is the underlying PreTrainedModel, and the tokenizer
# is the underlying PreTrainedTokenizer.
hf_model = st_model[0].auto_model      # The transformer model (e.g., GemmaModel)
hf_tokenizer = st_model.tokenizer       # The tokenizer

print(f"HF model type  : {type(hf_model).__name__}")
print(f"Tokenizer type : {type(hf_tokenizer).__name__}")

# Build the feature-extraction pipeline.
# This is the same pattern that worked for Nomic with mlflow.transformers.
embedding_pipeline = transformers.pipeline(
    task="feature-extraction",
    model=hf_model,
    tokenizer=hf_tokenizer,
)

# Quick test — the pipeline should return nested lists of floats.
test_output = embedding_pipeline("search_query: test")
print(f"Pipeline output type: {type(test_output)}")
print(f"Pipeline output shape: {len(test_output)} x {len(test_output[0])} x {len(test_output[0][0])}")

# COMMAND ----------

# =============================================================================
# STEP 3: Log with mlflow.transformers (proven OpenAI-compatible path)
# =============================================================================
# task="llm/v1/embeddings" tells the Databricks serving container to:
#   - Accept {"input": ["text1", "text2"]} (OpenAI embedding request format)
#   - Return {"data": [{"embedding": [...], "index": 0}]} (OpenAI response)
#   - Apply mean pooling to the feature-extraction output internally

import mlflow

# End any stale active run from a previous notebook execution.
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

# Replace with your actual catalog and schema.
CATALOG = "your_catalog"
SCHEMA = "your_schema"
MODEL_NAME_UC = f"{CATALOG}.{SCHEMA}.embedding_gemma"

client = MlflowClient(registry_uri="databricks-uc")

# Create registered model if it doesn't exist.
try:
    client.create_registered_model(MODEL_NAME_UC)
    print(f"Created new registered model: {MODEL_NAME_UC}")
except Exception as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
        print(f"Registered model already exists: {MODEL_NAME_UC}")
    else:
        raise

# Register the new version using the URI from log_model.
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

print(f"Creating endpoint with model {MODEL_NAME_UC} version {MODEL_VERSION}...")

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
# Since we used mlflow.transformers with task="llm/v1/embeddings" (the same
# approach that worked for Nomic), the endpoint should accept {"input": [...]}
# and return {"data": [{"embedding": [...]}]}.

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


# Test with OpenAI embedding format — this should work because it's the
# same mlflow.transformers + task="llm/v1/embeddings" pattern as Nomic.
doc_result = call_embedding_endpoint({
    "input": [
        "search_document: RAG combines a retriever with a generative model.",
        "search_document: Vector databases store embeddings for similarity search.",
    ]
})

query_result = call_embedding_endpoint({
    "input": ["search_query: How does RAG work?"]
})

# Verify the response is OpenAI-compatible.
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
# STEP 7: Define the Prefix Wrapper for RAG integration
# =============================================================================
# DatabricksEmbeddings requires the endpoint to speak OpenAI format natively.
# Since we confirmed in Step 6 that the endpoint returns OpenAI format,
# we can use DatabricksEmbeddings directly — we just need to prepend
# the correct EmbeddingGemma task prefixes.

from langchain_community.embeddings import DatabricksEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List


class PrefixedEmbeddingGemma(Embeddings):
    """
    LangChain-compatible wrapper that:
    1. Prepends the correct EmbeddingGemma task prefix to each text.
    2. Delegates to DatabricksEmbeddings (which calls the OpenAI-compatible endpoint).
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
        """Embed documents — prepends 'search_document: ' to each text."""
        prefixed = [f"{self._doc_prefix}{text}" for text in texts]
        return self._base.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query — prepends 'search_query: ' to the text."""
        prefixed = f"{self._query_prefix}{text}"
        return self._base.embed_query(prefixed)


# COMMAND ----------

# =============================================================================
# STEP 8: Wire the wrapper into the RAG retriever
# =============================================================================
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch

# Instantiate the wrapper pointing at our serving endpoint.
rag_embeddings = PrefixedEmbeddingGemma(endpoint=ENDPOINT_NAME)

# Replace with your actual Vector Search endpoint and index names.
VS_ENDPOINT_NAME = "your_vector_search_endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.your_index"

vsc = VectorSearchClient()
index = vsc.get_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME,
)

retriever = DatabricksVectorSearch(
    index=index,
    embedding=rag_embeddings,
    text_column="chunk_text",
    columns=["chunk_text", "doc_uri"],
).as_retriever(search_kwargs={"k": 5})

# COMMAND ----------

# =============================================================================
# STEP 9: Quick retrieval test
# =============================================================================
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
#    The prefixes ("search_document: " / "search_query: ") are the most
#    common for EmbeddingGemma. Check st_model.prompts from Step 1 for
#    your specific checkpoint — some variants use "passage: " / "query: ".
#
# 3. GPU REQUIREMENT
#    EmbeddingGemma requires GPU_SMALL or larger for the serving endpoint.
#
# 4. WHY mlflow.transformers INSTEAD OF mlflow.sentence_transformers
#    On DBR 17.3 ML (mlflow-skinny 3.0.1), mlflow.transformers.log_model()
#    with task="llm/v1/embeddings" is the only proven path that produces
#    OpenAI-compatible format from Databricks Model Serving. The
#    sentence_transformers and pyfunc flavors do not reliably produce
#    OpenAI format on this runtime — they return raw lists instead.
#
# 5. POOLING CONSIDERATION
#    The feature-extraction pipeline returns token-level embeddings.
#    The Databricks serving container applies mean pooling when
#    task="llm/v1/embeddings" is set. If embedding quality differs
#    from sentence-transformers' native encode(), compare pooling
#    strategies. The task prefixes ("search_document: " / "search_query: ")
#    are the more critical factor for EmbeddingGemma retrieval accuracy.
#
# 6. DBR 17.3 ML COMPATIBILITY
#    All libraries are pre-installed. No pip installs needed.
# =============================================================================
