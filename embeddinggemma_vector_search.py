# Databricks notebook source

# MAGIC %md
# MAGIC # EmbeddingGemma — Vector Search Index, DatabricksEmbeddings & LangGraph RAG
# MAGIC
# MAGIC **Runtime:** DBR 17.3 LTS ML (GPU)
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook builds on the EmbeddingGemma serving endpoint created in
# MAGIC `embeddinggemma_serving.py` and covers three additional steps:
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |------|-------------|
# MAGIC | 1 | Configuration — reuses endpoint names from the serving notebook |
# MAGIC | 2 | Prepare the source Delta table with Change Data Feed enabled |
# MAGIC | 3 | Create the Vector Search endpoint and Delta Sync index |
# MAGIC | 4 | Use `DatabricksEmbeddings` directly for query and document embedding |
# MAGIC | 5 | Build a LangGraph RAG agent that retrieves from the index and generates answers |
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - The EmbeddingGemma serving endpoints from `embeddinggemma_serving.py` must be
# MAGIC   running (both the query endpoint and the document endpoint).
# MAGIC - A source Delta table containing the documents you want to index.
# MAGIC - Serverless compute enabled in your workspace (required for Vector Search).
# MAGIC - `CREATE TABLE` privileges on the target catalog/schema.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Install Dependencies

# COMMAND ----------

# %pip install databricks-vectorsearch databricks-langchain langchain langgraph
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Configuration
# MAGIC
# MAGIC All configurable values in one place. Update these to match your environment.
# MAGIC The endpoint names must match the serving endpoints created in the serving notebook.

# COMMAND ----------

# -- Unity Catalog coordinates --
CATALOG = "your_catalog"
SCHEMA  = "your_schema"

# -- Source Delta table --
# Must have a unique primary key column and a text column to embed.
# The table must exist before running Step 3.
SOURCE_TABLE   = f"{CATALOG}.{SCHEMA}.your_documents_table"
PRIMARY_KEY    = "id"           # unique integer or string column
TEXT_COLUMN    = "content"      # column containing the document text to embed

# -- EmbeddingGemma serving endpoints (from embeddinggemma_serving.py) --
# Use the document endpoint for ingestion (Retrieval-document prompt mode).
# Use the query endpoint for retrieval at search time (Retrieval-query prompt mode).
DOCUMENT_ENDPOINT_NAME = "embeddinggemma-300m-document-endpoint"
QUERY_ENDPOINT_NAME    = "embeddinggemma-300m-endpoint"

# -- Vector Search --
VS_ENDPOINT_NAME = "embeddinggemma-vs-endpoint"
VS_INDEX_NAME    = f"{CATALOG}.{SCHEMA}.embeddinggemma_index"

# -- LLM endpoint for the LangGraph RAG agent (Step 5) --
# Any Databricks-hosted chat model endpoint that supports function calling.
# Replace with your preferred endpoint.
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Prepare the Source Delta Table
# MAGIC
# MAGIC The source Delta table must have **Change Data Feed** enabled. This allows
# MAGIC Vector Search to incrementally sync only rows that have changed since the last
# MAGIC sync, rather than reprocessing the entire table every time.
# MAGIC
# MAGIC Run this once. It is safe to re-run — `ALTER TABLE` with `SET TBLPROPERTIES`
# MAGIC is idempotent.

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {SOURCE_TABLE}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print(f"Change Data Feed enabled on {SOURCE_TABLE}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Create the Vector Search Endpoint and Index
# MAGIC
# MAGIC ### Vector Search endpoint vs. Model Serving endpoint
# MAGIC
# MAGIC These are two separate Databricks resources:
# MAGIC
# MAGIC - **Model Serving endpoint** — hosts EmbeddingGemma, called by Vector Search
# MAGIC   to compute embeddings during index creation and sync.
# MAGIC - **Vector Search endpoint** — stores and serves the vector index itself,
# MAGIC   handling ANN (approximate nearest neighbor) search queries.
# MAGIC
# MAGIC ### Sync modes
# MAGIC
# MAGIC - `TRIGGERED` — you call `index.sync()` manually to update the index. Lower
# MAGIC   cost; suitable for batch pipelines where documents are added periodically.
# MAGIC - `CONTINUOUS` — the index stays in sync with seconds of latency via a
# MAGIC   streaming pipeline. Higher cost; suitable for near-real-time use cases.
# MAGIC
# MAGIC ### Scale to zero note
# MAGIC
# MAGIC Databricks recommends disabling Scale to Zero on the embedding endpoint used
# MAGIC for Vector Search ingestion. If the endpoint is scaled down when a sync starts,
# MAGIC the first request may time out. You can update this in the Serving UI.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()  # auto-detects credentials inside a Databricks notebook

# ---- Create Vector Search endpoint if it doesn't exist ----
if not vsc.endpoint_exists(VS_ENDPOINT_NAME):
    vsc.create_endpoint_and_wait(
        name=VS_ENDPOINT_NAME,
        endpoint_type="STANDARD",
    )
    print(f"Vector Search endpoint '{VS_ENDPOINT_NAME}' created.")
else:
    print(f"Vector Search endpoint '{VS_ENDPOINT_NAME}' already exists.")

# ---- Create Delta Sync index if it doesn't exist ----
if not vsc.index_exists(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME):
    index = vsc.create_delta_sync_index_and_wait(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=SOURCE_TABLE,
        pipeline_type="TRIGGERED",              # change to "CONTINUOUS" for near-real-time
        primary_key=PRIMARY_KEY,
        embedding_source_column=TEXT_COLUMN,
        # Document endpoint used for ingestion — uses "Retrieval-document" prompt mode
        embedding_model_endpoint_name=DOCUMENT_ENDPOINT_NAME,
        # Query endpoint used at search time — uses "Retrieval-query" prompt mode.
        # This is the correct split for EmbeddingGemma: different prompt prefixes
        # are applied automatically by sentence-transformers based on the endpoint's
        # default prompt_name.
        model_endpoint_name_for_query=QUERY_ENDPOINT_NAME,
    )
    print(f"Index '{VS_INDEX_NAME}' created and initial sync complete.")
else:
    index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
    print(f"Index '{VS_INDEX_NAME}' already exists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Trigger a Manual Sync (TRIGGERED mode only)
# MAGIC
# MAGIC If you added or updated rows in the source table and want to sync the index,
# MAGIC call `index.sync()`. Skip this if you are using `CONTINUOUS` sync mode.

# COMMAND ----------

index.sync()
print("Index sync triggered. Check the Vector Search UI for progress.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Direct Similarity Search via the Vector Search SDK
# MAGIC
# MAGIC The simplest way to query the index — no LangChain required. Useful for
# MAGIC testing the index and for pipelines that don't need an LLM in the loop.

# COMMAND ----------

results = index.similarity_search(
    query_text="What is machine learning?",
    columns=[PRIMARY_KEY, TEXT_COLUMN],
    num_results=5,
)

print("Top results:")
for row in results["result"]["data_array"]:
    print(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 — DatabricksEmbeddings
# MAGIC
# MAGIC `DatabricksEmbeddings` from the `databricks-langchain` package wraps any
# MAGIC OpenAI-compatible Databricks Model Serving endpoint as a LangChain
# MAGIC `Embeddings` object. Because your EmbeddingGemma endpoint returns
# MAGIC OpenAI-format responses, it works directly.
# MAGIC
# MAGIC Two separate instances are created — one per prompt mode — mirroring the
# MAGIC two-endpoint design from the serving notebook:
# MAGIC
# MAGIC - **`query_embeddings`** — for embedding user queries at retrieval time.
# MAGIC   Calls the endpoint whose `predict` defaults to `"Retrieval-query"`.
# MAGIC - **`document_embeddings`** — for embedding documents during indexing.
# MAGIC   Calls the endpoint whose `predict` defaults to `"Retrieval-document"`.

# COMMAND ----------

from databricks_langchain import DatabricksEmbeddings

# For embedding user queries (Retrieval-query prompt mode)
query_embeddings = DatabricksEmbeddings(endpoint=QUERY_ENDPOINT_NAME)

# For embedding documents (Retrieval-document prompt mode)
document_embeddings = DatabricksEmbeddings(endpoint=DOCUMENT_ENDPOINT_NAME)

# ---- Embed a single query ----
query_vector = query_embeddings.embed_query("What is machine learning?")
print(f"Query embedding dimension: {len(query_vector)}")
print(f"First 5 values: {query_vector[:5]}")

# ---- Embed a batch of documents ----
docs = [
    "Machine learning is a subset of artificial intelligence.",
    "Neural networks are inspired by the human brain.",
    "Transformers are the dominant architecture for NLP tasks.",
]
doc_vectors = document_embeddings.embed_documents(docs)
print(f"\nEmbedded {len(doc_vectors)} documents, dimension: {len(doc_vectors[0])}")

# ---- Async usage (useful in async notebook cells or FastAPI routes) ----
import asyncio

async def async_example():
    vector = await query_embeddings.aembed_query("What is deep learning?")
    print(f"Async query embedding dimension: {len(vector)}")

asyncio.run(async_example())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8 — DatabricksVectorSearch as a LangChain Retriever
# MAGIC
# MAGIC `DatabricksVectorSearch` wraps the Vector Search index as a LangChain
# MAGIC `VectorStore`, giving access to `.as_retriever()` which returns a standard
# MAGIC LangChain `Retriever`. This is the bridge between the Vector Search index
# MAGIC and LangChain / LangGraph pipelines.
# MAGIC
# MAGIC We pass `query_embeddings` here because `DatabricksVectorSearch` uses it to
# MAGIC embed the query text before searching the index.

# COMMAND ----------

from databricks_langchain import DatabricksVectorSearch

vector_store = DatabricksVectorSearch(
    index_name=VS_INDEX_NAME,
    endpoint=VS_ENDPOINT_NAME,
    embedding=query_embeddings,     # used to embed queries at retrieval time
    text_column=TEXT_COLUMN,        # which column to return as document content
    columns=[PRIMARY_KEY, TEXT_COLUMN],  # columns to return in search results
)

# Create a LangChain retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Test the retriever directly
retrieved_docs = retriever.invoke("What is machine learning?")
print(f"Retrieved {len(retrieved_docs)} documents:")
for doc in retrieved_docs:
    print(f"  - {doc.page_content[:120]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9 — LangGraph RAG Agent
# MAGIC
# MAGIC A LangGraph RAG agent that:
# MAGIC
# MAGIC 1. Takes a user question as input.
# MAGIC 2. Retrieves relevant documents from the Vector Search index using EmbeddingGemma.
# MAGIC 3. Generates an answer using a Databricks-hosted LLM, grounded in the
# MAGIC    retrieved context.
# MAGIC
# MAGIC ### Graph structure
# MAGIC
# MAGIC ```
# MAGIC  [START] → retrieve → generate → [END]
# MAGIC ```
# MAGIC
# MAGIC This is a simple two-node graph. It can be extended with additional nodes
# MAGIC for query rewriting, grading retrieved documents, or routing to different
# MAGIC retrieval strategies.
# MAGIC
# MAGIC ### State
# MAGIC
# MAGIC LangGraph passes a shared `State` dict between nodes. Each node reads from
# MAGIC and writes to this dict. The state here carries:
# MAGIC - `question` — the user's input question
# MAGIC - `context`  — the documents retrieved from Vector Search
# MAGIC - `answer`   — the LLM's generated answer

# COMMAND ----------

from typing import List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from databricks_langchain import ChatDatabricks
from langgraph.graph import START, END, StateGraph


# ---- State definition ----

class RAGState(TypedDict):
    question: str
    context:  List[Document]
    answer:   str


# ---- LLM ----

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT_NAME,
    temperature=0.1,
    max_tokens=512,
)


# ---- Prompt ----

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful assistant. Answer the user's question using only "
            "the provided context. If the context does not contain enough information "
            "to answer the question, say so clearly. Do not make up information.\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{question}"),
])


# ---- Node: retrieve ----

def retrieve(state: RAGState) -> RAGState:
    """
    Retrieve documents from the Vector Search index relevant to the question.
    Uses EmbeddingGemma (Retrieval-query prompt mode) via DatabricksVectorSearch.
    """
    docs = retriever.invoke(state["question"])
    return {"context": docs}


# ---- Node: generate ----

def generate(state: RAGState) -> RAGState:
    """
    Generate an answer from the LLM, grounded in the retrieved context.
    """
    # Concatenate retrieved document content into a single context string
    context_str = "\n\n".join(doc.page_content for doc in state["context"])

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context":  context_str,
        "question": state["question"],
    })
    return {"answer": answer}


# ---- Build the graph ----

graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

rag_graph = graph_builder.compile()

print("LangGraph RAG agent compiled successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10 — Run the RAG Agent
# MAGIC
# MAGIC Invoke the compiled graph with a question. The graph runs `retrieve` then
# MAGIC `generate` and returns the final state, which includes the question, the
# MAGIC retrieved context documents, and the generated answer.

# COMMAND ----------

def ask(question: str) -> str:
    """Run the RAG agent and return the answer."""
    result = rag_graph.invoke({"question": question})
    return result["answer"]


# ---- Single question ----
answer = ask("What is machine learning?")
print(f"Answer:\n{answer}")

# COMMAND ----------

# ---- Batch questions ----
questions = [
    "What is machine learning?",
    "How do neural networks work?",
    "What are transformers used for?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask(q)}")
    print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11 — Inspect Intermediate State (for debugging)
# MAGIC
# MAGIC Stream the graph execution to inspect what was retrieved before the answer
# MAGIC is generated. Useful for debugging retrieval quality.

# COMMAND ----------

question = "What is machine learning?"

for step in rag_graph.stream({"question": question}):
    node_name = list(step.keys())[0]
    node_output = step[node_name]
    print(f"\n--- Node: {node_name} ---")

    if node_name == "retrieve":
        docs = node_output.get("context", [])
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.page_content[:150]}...")

    elif node_name == "generate":
        print(f"Answer: {node_output.get('answer', '')}")
