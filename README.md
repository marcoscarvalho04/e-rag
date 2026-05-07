# 🧠 e-rag

**Enhanced RAG** — retrieval augmented generation with semantic context preservation.

Classic RAG treats chunks as isolated vectors. e-rag builds a semantic affinity graph between chunks and uses community detection to return coherent regions of knowledge — not isolated fragments.

---

## 🧱 The problem with classic RAG

```
document: A → B → C → D  (sentences with dependencies)
classic:  {A}, {B}, {C}, {D}  (independent chunks)
retrieval: returns C  →  but C without B makes no sense
```

When you cut text into fixed-size chunks, you destroy the context that gives each piece meaning. **The chunk is a brick, and the mortar — the semantic context — is lost.**

---

## ✨ How e-rag solves it

🔪 **Semantic chunking** — instead of fixed sizes, boundaries are detected by measuring semantic drift between sentences via a sliding window. Chunks are cut where ideas actually change.

🕸️ **K-NN graph** — after embedding, each chunk connects to its K most semantically similar neighbors. The graph makes explicit what classic RAG leaves implicit.

🔍 **Louvain community detection** — the graph's natural communities emerge without any arbitrary threshold. The algorithm finds the structure that already exists in the data.

🎯 **Community-aware retrieval** — a query returns not just the nearest chunk, but the coherent semantic neighborhood it belongs to. The LLM receives context, not fragments.

```
query → nearest chunk (anchor) → its community → ranked by relevance
```

---

## 📦 Install

```bash
go get github.com/marcoscarvalho04/e-rag
```

> Requires Go 1.22+

---

## 🚀 Usage

```go
import "github.com/marcoscarvalho04/e-rag/pkg/erag"

// define any embedding function
embedder := func(texts []string) ([][]float32, error) {
    // use Voyage AI, OpenAI, Ollama, a local ONNX model — anything
}

// build index from documents
idx, err := erag.New([]string{"document 1...", "document 2..."}, embedder)

// search
result, err := idx.Search("what is the impact of interest rates?", 5)

fmt.Println(result.Anchor.Text)   // most relevant chunk
fmt.Println(result.Anchor.Score)  // cosine similarity

for _, c := range result.Community {
    fmt.Println(c.Score, c.Text)  // semantic neighbors, ranked
}
```

### 💾 Persist and load

```go
// save index to disk — embedding calls happen once, at index time
if err := idx.Save("knowledge.gob"); err != nil {
    log.Fatal(err)
}

// load and serve — no API calls on startup
idx, err := erag.Load("knowledge.gob", embedder)
```

### ⚙️ Options

```go
idx, err := erag.New(docs, embedder,
    erag.WithK(10),                  // K-NN neighbors per chunk (default: 5)
    erag.WithWindowSize(4),          // sliding window size (default: 3)
    erag.WithBreakThreshold(0.3),    // boundary sensitivity (default: 0.25)
)
```

| Option | Description | Default |
|--------|-------------|---------|
| `WithK(k)` | Neighbors per chunk in the graph. Higher = more relationships, higher indexing cost | `5` |
| `WithWindowSize(n)` | Sentences averaged on each side of a boundary candidate. Larger = smoother detection | `3` |
| `WithBreakThreshold(t)` | How sharp a semantic drop must be to become a chunk boundary. Higher = smaller chunks | `0.25` |

---

## 🔌 Embedder examples

e-rag is **embedder-agnostic**. Any function with the signature `func([]string) ([][]float32, error)` works.

<details>
<summary><b>Voyage AI</b></summary>

```go
func voyageEmbedder(texts []string) ([][]float32, error) {
    // https://www.voyageai.com
    // model: voyage-3-lite
}
```
</details>

<details>
<summary><b>OpenAI</b></summary>

```go
func openAIEmbedder(texts []string) ([][]float32, error) {
    // model: text-embedding-3-small
}
```
</details>

<details>
<summary><b>Ollama (local, zero cost)</b></summary>

```go
func ollamaEmbedder(texts []string) ([][]float32, error) {
    // model: nomic-embed-text
    // runs fully local, no API key needed
}
```
</details>

---

## 🔬 Comparison demo

The `examples/comparison/` directory runs a side-by-side comparison between classic RAG and e-rag using the original RAG paper (Lewis et al., 2020) as corpus.

```bash
VOYAGE_API_KEY=your-key go run ./examples/comparison/
```

Sample output for the query *"how does index hot-swapping work without retraining?"*:

```
📦 CLASSIC RAG  (top-3 by cosine similarity)
  [1] score=0.6356  Index Hot-Swapping  Updating the knowledge index...
  [2] score=0.4128  Training Approach  The model is trained end-to-end...
  [3] score=0.3362  Ablation Studies  Retrieval Learning  Freezing the retriever...

🧠 E-RAG  (anchor + semantic community)
  anchor  score=0.6356  community=1
      Index Hot-Swapping  Updating the knowledge index...
  community neighbors (same semantic region):
    · score=0.4128  Training Approach  The model is trained end-to-end...
    · score=0.3362  Ablation Studies  Retrieval Learning  Freezing the retriever...
    · score=0.2990  The approach combines advantages of hybrid memory systems...
```

Both return the same anchor chunk. The difference: e-rag knows that anchor belongs to **community 1** — the semantic region covering how the model trains and behaves post-indexing. A follow-up query in the same session can navigate this community directly, and an LLM receives a coherent narrative region instead of isolated cosine-ranked fragments.

---

## 🛠️ CLI tools

The repository includes two CLI tools as usage examples.

**Index a document**

```bash
go run ./cmd/erag-index/ -input document.txt -output index.gob -k 5
```

**Serve over HTTP**

```bash
VOYAGE_API_KEY=your-key go run ./cmd/erag-serve/
```

```bash
# create index
curl -X POST http://localhost:8080/indexes/myindex \
  -H "Content-Type: application/json" \
  -d '{"documents": ["text one...", "text two..."], "k": 5}'

# search
curl -X POST http://localhost:8080/indexes/myindex/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "top_k": 5}'
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/indexes` | List all indexes |
| `POST` | `/indexes/{name}` | Create or update index |
| `DELETE` | `/indexes/{name}` | Delete index |
| `POST` | `/indexes/{name}/search` | Search index |

> The HTTP server is a reference implementation. For production use, wrap `pkg/erag` directly in your application.

---

## ⚡ Performance

Benchmarks on AMD Ryzen 7 5700G, 512-dimension embeddings, `go test -bench -benchtime=3s`.

**Indexing** (`New`) — runs once, offline:

| Documents | Time |
|-----------|------|
| 10 | 0.5ms |
| 50 | 5ms |
| 100 | 15ms |
| 500 | 126ms |

> Indexing is O(n²) due to K-NN graph construction. Suitable for corpora up to ~10,000 chunks. HNSW optimization is planned for larger corpora.

**Retrieval** (`Search`) — runs on every query:

| Indexed chunks | Latency |
|----------------|---------|
| 10 | 10µs |
| 50 | 36µs |
| 100 | **72µs** |
| 500 | 242µs |

**Persistence:**

| Operation | Time |
|-----------|------|
| Save (100 chunks) | ~2ms |
| Load (100 chunks) | ~1.3ms |

---

## 🔬 How it compares

| | Classic RAG | GraphRAG (Microsoft) | **e-rag** |
|---|---|---|---|
| Chunk boundary | Fixed size | Fixed size | ✅ Semantic |
| Context preservation | ❌ None | LLM-generated | ✅ Graph structure |
| LLM calls at index time | None | ❌ O(n) — expensive | ✅ None |
| Retrieval unit | Isolated chunk | Community | ✅ Community |
| Cost | Low | High | ✅ Low |

> GraphRAG uses an LLM to generate context for each chunk at indexing time — that's n LLM calls for n chunks. e-rag derives the same community structure mathematically from the embeddings, with zero extra API cost.

---

## 🏗️ Architecture

```
documents
    ↓
semantic chunking       sliding window detects topic shifts between sentences
    ↓
embeddings              any model — one batch call per document set
    ↓
K-NN graph              each chunk connects to K nearest neighbors by cosine similarity
    ↓
Louvain communities     natural clusters emerge without arbitrary thresholds
    ↓
persisted index         chunks + graph + communities serialized to disk (gob)

query
    ↓
embed query
    ↓
graph search            finds anchor chunk via cosine similarity
    ↓
community retrieval     returns anchor + semantic neighbors, ranked by relevance
```

---

## 📄 License

MIT
