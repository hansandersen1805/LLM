# Databricks notebook source

# MAGIC %md
# MAGIC # EmbeddingGemma — OpenAI-Compatible Model Serving Endpoint
# MAGIC
# MAGIC **Runtime:** DBR 17.3 LTS ML (GPU)
# MAGIC
# MAGIC ## Background
# MAGIC
# MAGIC EmbeddingGemma requires `sentence-transformers>=5.0.0` and `transformers>=4.56.0`,
# MAGIC but DBR 17.3 ML ships with `sentence-transformers==4.0.1` and `transformers==4.51.3`.
# MAGIC Upgrading on the cluster risks breaking other pre-installed packages.
# MAGIC
# MAGIC **Strategy:** We wrap EmbeddingGemma in an `mlflow.pyfunc.PythonModel` and declare
# MAGIC the newer package versions in `pip_requirements`. MLflow saves these requirements
# MAGIC alongside the model artifacts. When Databricks Model Serving creates the serving
# MAGIC container, it installs those packages into the container — not on the cluster.
# MAGIC The result is an OpenAI-compatible `/v1/embeddings` endpoint that works with
# MAGIC `DatabricksEmbedding`, while the cluster stays untouched.
# MAGIC
# MAGIC ## How `mlflow.pyfunc` and `sentence_transformers` Work Together
# MAGIC
# MAGIC This notebook uses **two frameworks** that play different roles at different stages:
# MAGIC
# MAGIC | Framework | When it runs | Where it runs | What it does |
# MAGIC |-----------|-------------|---------------|-------------|
# MAGIC | `mlflow.pyfunc` | **Log time** (Step 4) | Your cluster | Packages the PyFunc wrapper class, model artifacts (weights, config, tokenizer), and `requirements.txt` into a versioned MLflow model in Unity Catalog. It does **not** import or execute `sentence_transformers`. |
# MAGIC | `sentence_transformers` | **Serve time** (Steps 5–7) | Serving container | Installed from `pip_requirements` when the container starts. The PyFunc wrapper's `load_context` method imports `SentenceTransformer` and loads EmbeddingGemma. The `predict` method calls `model.encode(texts, prompt_name=...)`, which automatically applies the correct prompt prefixes, tokenizes, pools, normalizes, and returns embeddings. |
# MAGIC
# MAGIC This separation is what makes the approach work: the cluster only needs to
# MAGIC serialize the Python class and copy the model files (both possible with the
# MAGIC pre-installed packages). The serving container independently installs
# MAGIC `sentence-transformers>=5.0.0` and handles all the EmbeddingGemma-specific logic.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - EmbeddingGemma model files already downloaded to ADLS and accessible via a Unity Catalog Volume.
# MAGIC - A GPU-enabled cluster running DBR 17.3 LTS ML.
# MAGIC - Permissions to register models in Unity Catalog and create serving endpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Configuration
# MAGIC
# MAGIC Define all configurable values in one place: the UC Volume path where
# MAGIC EmbeddingGemma is stored, the Unity Catalog location for registering the
# MAGIC model, and the serving endpoint name. Update these to match your environment.
# MAGIC
# MAGIC Note: In MLflow 3.x (which ships as `mlflow-skinny==3.0.1` on DBR 17.3 ML),
# MAGIC the default model registry URI is already `databricks-uc`, so an explicit
# MAGIC `mlflow.set_registry_uri()` call is no longer required.

# COMMAND ----------

# -- Unity Catalog coordinates --
CATALOG = "your_catalog"
SCHEMA = "your_schema"
MODEL_NAME = "embeddinggemma_300m"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# -- Path to EmbeddingGemma on your UC Volume --
# This is the Volume path where you downloaded the model files from ADLS.
# It should contain config.json, model.safetensors, tokenizer files, etc.
VOLUME_MODEL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/your_volume/embeddinggemma-300m"

# -- Serving endpoint --
ENDPOINT_NAME = "embeddinggemma-300m-endpoint"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Verify the Model Files Exist on the Volume
# MAGIC
# MAGIC A quick sanity check to confirm the EmbeddingGemma files are accessible
# MAGIC at the expected path before we attempt to log them with MLflow. This
# MAGIC avoids a confusing error later if the path is wrong or permissions are missing.

# COMMAND ----------

import os

volume_contents = os.listdir(VOLUME_MODEL_PATH)
print(f"Files found at {VOLUME_MODEL_PATH}:")
for f in sorted(volume_contents):
    print(f"  {f}")

# Basic validation — these files must be present for sentence-transformers to load the model
required_files = ["config.json"]
missing = [f for f in required_files if f not in volume_contents]
if missing:
    raise FileNotFoundError(
        f"Missing expected files in {VOLUME_MODEL_PATH}: {missing}. "
        "Check that the full EmbeddingGemma model was downloaded."
    )
print("\nModel files verified.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Define the PyFunc Wrapper
# MAGIC
# MAGIC This is the core of the approach. We define a custom `mlflow.pyfunc.PythonModel`
# MAGIC subclass. **Important:** this class is only *defined* on the cluster — it is not
# MAGIC *executed* here. MLflow serializes it via `cloudpickle` when we call `log_model`
# MAGIC in Step 4. The actual execution happens inside the serving container, where the
# MAGIC correct dependencies are installed.
# MAGIC
# MAGIC The class has two methods:
# MAGIC
# MAGIC 1. **`load_context`** — Called once when the serving container starts. It imports
# MAGIC    `sentence_transformers.SentenceTransformer` (available in the container because
# MAGIC    `pip_requirements` installs `>=5.0.0`) and loads EmbeddingGemma from the model
# MAGIC    artifacts that MLflow copied from the UC Volume. This is the only place
# MAGIC    `sentence_transformers` is used — the cluster never imports it for this model.
# MAGIC
# MAGIC 2. **`predict`** — Called on every inference request. It:
# MAGIC    - Parses the input from Databricks Model Serving (a DataFrame with an `"input"` column).
# MAGIC    - Calls `self.model.encode(texts, prompt_name="Retrieval-query")`. This is a
# MAGIC      `sentence_transformers` method that automatically prepends the correct
# MAGIC      EmbeddingGemma prompt prefix (`"task: search result | query: "` for queries)
# MAGIC      before tokenizing, encoding, pooling, and normalizing.
# MAGIC    - Formats the result into the OpenAI `/v1/embeddings` JSON schema so that
# MAGIC      `DatabricksEmbedding` can consume it directly.

# COMMAND ----------

import mlflow
import numpy as np


class EmbeddingGemmaModel(mlflow.pyfunc.PythonModel):
    """
    PyFunc wrapper that loads EmbeddingGemma via sentence-transformers
    and returns OpenAI-compatible embedding responses.

    Architecture note:
        This class is DEFINED on the cluster but EXECUTED in the serving container.
        At log time (Step 4), MLflow serializes this class and stores it alongside the
        model artifacts. At serve time (Step 5+), the serving container deserializes it,
        installs the packages from pip_requirements (including sentence-transformers>=5.0.0),
        and then calls load_context() and predict().

        The cluster never needs to import sentence_transformers for EmbeddingGemma —
        all sentence_transformers usage is deferred to the serving container via
        lazy imports inside load_context() and predict().
    """

    def load_context(self, context):
        """
        Called once when the serving container starts. Loads the model
        from the artifacts directory using sentence-transformers.

        This is where sentence_transformers is first imported. The import
        succeeds because the serving container installed sentence-transformers>=5.0.0
        from the pip_requirements declared in Step 4. On the cluster (which only has
        4.0.1), this method is never called — the class is only serialized, not executed.
        """
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            context.artifacts["model_dir"],
            trust_remote_code=True,
        )
        self.model_name = "embeddinggemma-300m"
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def predict(self, context, model_input, params=None):
        """
        Encode input texts and return an OpenAI-compatible response.

        Input format (from Databricks Model Serving):
            DataFrame with an "input" column containing strings,
            or a dict with {"input": ["text1", "text2", ...]}.

        Output format (OpenAI /v1/embeddings):
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [...]},
                    ...
                ],
                "model": "embeddinggemma-300m",
                "usage": {"prompt_tokens": N, "total_tokens": N}
            }
        """
        import pandas as pd

        # ---- Extract input texts ----
        if isinstance(model_input, pd.DataFrame):
            if "input" in model_input.columns:
                texts = model_input["input"].tolist()
            else:
                texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, dict):
            texts = model_input.get("input", [])
        elif isinstance(model_input, list):
            texts = model_input
        else:
            texts = [str(model_input)]

        if isinstance(texts, str):
            texts = [texts]

        # ---- Determine which prompt template to apply ----
        # Default is "Retrieval-query" which prepends "task: search result | query: "
        # For document indexing, use "Retrieval-document" which prepends "title: none | text: "
        prompt_name = "Retrieval-query"
        if params and isinstance(params, dict):
            prompt_name = params.get("prompt_name", prompt_name)

        # ---- Encode ----
        embeddings = self.model.encode(
            texts,
            prompt_name=prompt_name,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # ---- Approximate token count ----
        total_tokens = sum(len(t.split()) for t in texts)

        # ---- Build OpenAI-compatible response ----
        data = [
            {
                "object": "embedding",
                "index": idx,
                "embedding": emb.tolist(),
            }
            for idx, emb in enumerate(embeddings)
        ]

        return {
            "object": "list",
            "data": data,
            "model": self.model_name,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Log the Model to Unity Catalog
# MAGIC
# MAGIC This step registers the PyFunc wrapper and the model files as an MLflow model
# MAGIC in Unity Catalog. **No `sentence_transformers` code is executed here.** MLflow
# MAGIC simply:
# MAGIC
# MAGIC 1. Serializes the `EmbeddingGemmaModel` class definition via `cloudpickle`.
# MAGIC 2. Copies the model artifact files from the UC Volume into the MLflow artifact store.
# MAGIC 3. Saves the `pip_requirements` as a `requirements.txt` alongside the artifacts.
# MAGIC 4. Registers a new model version in Unity Catalog.
# MAGIC
# MAGIC The two key parameters that bridge the pyfunc ↔ sentence_transformers gap:
# MAGIC
# MAGIC - **`artifacts`** — Points MLflow to the UC Volume path containing the
# MAGIC   EmbeddingGemma files (weights, config, tokenizer). MLflow copies these so
# MAGIC   they travel with the registered model version. At serve time, `load_context`
# MAGIC   receives this path via `context.artifacts["model_dir"]` and passes it to
# MAGIC   `SentenceTransformer(...)`.
# MAGIC
# MAGIC - **`pip_requirements`** — Declares the Python packages the serving container
# MAGIC   needs. This is where the newer `sentence-transformers>=5.0.0` and
# MAGIC   `transformers>=4.56.0` are specified. When Databricks Model Serving builds
# MAGIC   the container, it reads this file and installs exactly these packages —
# MAGIC   your cluster is never affected.
# MAGIC
# MAGIC **MLflow 3.x syntax notes:**
# MAGIC - We use `name` instead of the deprecated `artifact_path` parameter.
# MAGIC - `mlflow.start_run()` is no longer required — in MLflow 3, models are
# MAGIC   first-class entities and can be logged directly. We include it here
# MAGIC   only to group related metadata under a single run for traceability.

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

# Signature tells the serving infrastructure what input/output to expect
input_schema = Schema([ColSpec("string", "input")])
output_schema = Schema([ColSpec("string")])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Example input for documentation and validation
input_example = {"input": ["What is machine learning?"]}

with mlflow.start_run(run_name="embeddinggemma-300m-pyfunc") as run:
    model_info = mlflow.pyfunc.log_model(
        # MLflow 3.x: use "name" instead of the deprecated "artifact_path"
        name="embeddinggemma-300m",
        python_model=EmbeddingGemmaModel(),
        # Points to the model files on the UC Volume — MLflow will copy them
        artifacts={"model_dir": VOLUME_MODEL_PATH},
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
        # These packages are installed in the SERVING CONTAINER only
        pip_requirements=[
            "sentence-transformers>=5.0.0",
            "transformers>=4.56.0",
            "torch>=2.3.0",
            "mlflow>=2.12.0",
            "numpy",
            "pandas",
        ],
    )

    # MLflow 3.x: model_uri uses models:/<model_id> format
    print(f"Model URI:     {model_info.model_uri}")
    print(f"Model ID:      {model_info.model_id}")
    print(f"Run ID:        {run.info.run_id}")
    print(f"Registered as: {REGISTERED_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Create the Model Serving Endpoint
# MAGIC
# MAGIC Uses the Databricks SDK to create a GPU-enabled serving endpoint that
# MAGIC hosts the model registered in Step 4. When the endpoint starts:
# MAGIC
# MAGIC 1. Databricks provisions a serving container with a GPU.
# MAGIC 2. The container installs the packages from the `requirements.txt` that
# MAGIC    MLflow saved in Step 4 (including `sentence-transformers>=5.0.0`).
# MAGIC 3. It calls `load_context` on our PyFunc wrapper to load EmbeddingGemma.
# MAGIC 4. The endpoint is ready to accept OpenAI-format embedding requests.
# MAGIC
# MAGIC `scale_to_zero_enabled=True` means the endpoint shuts down when idle to
# MAGIC save cost, but the first request after idle will have a cold-start delay.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

# Find the latest registered model version
latest_version = max(
    w.model_versions.list(REGISTERED_MODEL_NAME),
    key=lambda v: int(v.version),
).version
print(f"Deploying model version: {latest_version}")

# Create or update the endpoint
try:
    endpoint = w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=REGISTERED_MODEL_NAME,
                    entity_version=str(latest_version),
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                    workload_type="GPU_SMALL",
                ),
            ]
        ),
    )
    print(f"Endpoint '{ENDPOINT_NAME}' created successfully.")
    print(f"State: {endpoint.state}")

except Exception as e:
    if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"Endpoint '{ENDPOINT_NAME}' already exists — updating config...")
        w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME,
            served_entities=[
                ServedEntityInput(
                    entity_name=REGISTERED_MODEL_NAME,
                    entity_version=str(latest_version),
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                    workload_type="GPU_SMALL",
                ),
            ],
        )
        print("Endpoint updated successfully.")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Test the Endpoint
# MAGIC
# MAGIC Sends a test request to the serving endpoint to confirm it returns a
# MAGIC well-formed OpenAI-compatible embedding response. This verifies the full
# MAGIC chain: serving container → package install → model load → prompt handling → response format.

# COMMAND ----------

import requests
import json

# Get auth token and workspace URL
token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .apiToken()
    .get()
)
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

response = requests.post(
    f"https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json={
        "input": [
            "What is machine learning?",
            "Mars is the fourth planet from the Sun.",
        ]
    },
)

result = response.json()
print(f"HTTP status:          {response.status_code}")
print(f"Response object type: {result.get('object')}")
print(f"Number of embeddings: {len(result.get('data', []))}")
if result.get("data"):
    print(f"Embedding dimension:  {len(result['data'][0]['embedding'])}")
    print(f"Usage:                {result.get('usage')}")
else:
    print(f"Response body:\n{json.dumps(result, indent=2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — Use with DatabricksEmbedding
# MAGIC
# MAGIC Once the endpoint is running, you can point `DatabricksEmbedding` at it.
# MAGIC The endpoint defaults to `"Retrieval-query"` prompts, which is the correct
# MAGIC mode for query-time embeddings in a vector search pipeline.
# MAGIC
# MAGIC ```python
# MAGIC from databricks.vector_search.embedding import DatabricksEmbedding
# MAGIC
# MAGIC embedding_model = DatabricksEmbedding(endpoint=ENDPOINT_NAME)
# MAGIC embeddings = embedding_model.embed_query("What is machine learning?")
# MAGIC ```
# MAGIC
# MAGIC ### Query vs. Document Prompts
# MAGIC
# MAGIC EmbeddingGemma performs best when queries and documents use different prompts:
# MAGIC - **Queries** → `"Retrieval-query"` → prepends `"task: search result | query: "`
# MAGIC - **Documents** → `"Retrieval-document"` → prepends `"title: none | text: "`
# MAGIC
# MAGIC This endpoint defaults to query mode. For document indexing (e.g., when building
# MAGIC a vector search index), the simplest approach is to register a **second endpoint**
# MAGIC with the PyFunc wrapper's default `prompt_name` changed to `"Retrieval-document"`
# MAGIC in Step 3. This keeps each endpoint single-purpose and avoids relying on
# MAGIC `DatabricksEmbedding` to forward custom parameters.
