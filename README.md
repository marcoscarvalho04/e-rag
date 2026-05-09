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

## 🎯 Live showcase — where classic RAG breaks

The `examples/showcase/` directory indexes **three completely different domains** together — an ML paper, a sports science text, and a finance document — then runs ambiguous queries that could plausibly belong to any of them.

```bash
VOYAGE_API_KEY=your-key ANTHROPIC_API_KEY=your-key go run ./examples/showcase/
```

### The smoking gun

Query: **"what is the best long-term strategy to maximize performance?"**

The word *performance* appears in all three domains with incompatible meanings. Here is what each approach retrieves and what Claude generates from that context:

```
📦 CLASSIC RAG — top-3 by cosine similarity

  [1] [Finance / Index Funds]    score=0.5738
      Long-Term Strategy and Compound Performance — Index investing is
      fundamentally a long-term strategy...

  [2] [Finance / Index Funds]    score=0.3968
      The model for long-term wealth accumulation through index funds
      requires discipline during periods of poor performance...

  [3] [Sports / Periodization]   score=0.3691
      Periodization as an Optimization Model — Modern sports science
      views periodization as an optimization model...
```

```
🧠 E-RAG — anchor + semantic community

  anchor  [Finance / Index Funds]  score=0.5738  community=2
          Long-Term Strategy and Compound Performance...
  community:
    · [Finance / Index Funds]  score=0.3968  The model for long-term wealth...
    · [Finance / Index Funds]  score=0.3652  Risk Management and Portfolio...
    · [Finance / Index Funds]  score=0.3216  Index Funds and Portfolio Management...
```

Classic RAG scores by cosine and picks up a sports chunk as result #3 — it has no way to know the query already landed in a financial context. E-rag's Louvain communities isolated the finance domain completely: community 2 contains only finance chunks.

### LLM answers with each context

**With Classic RAG context** (mixed finance + sports):

> *"The best long-term strategy to maximize performance **depends on the domain**. For index investing, it involves maintaining a globally diversified portfolio... For athletic performance, there is no universal solution — successful coaches develop individualized periodization models..."*

**With e-rag context** (coherent finance community):

> *"The best long-term strategy to maximize performance is to maintain a globally diversified portfolio of index funds across asset classes, geographies, and time horizons. This requires discipline to avoid trading during market downturns, as investors who sell during poor performance lock in losses and forfeit subsequent recoveries. Regular rebalancing maintains target allocations, allowing compound growth to operate uninterrupted over multi-decade horizons."*

Classic RAG handed the LLM an incoherent context — two incompatible domains in the same prompt — and the model did the only reasonable thing: hedge. E-rag handed it a coherent semantic community and the model delivered a precise, actionable answer.

---

## 🔬 Comparison demo

The `examples/comparison/` directory runs a focused retrieval comparison using the original RAG paper (Lewis et al., 2020) as a single-domain corpus.

```bash
VOYAGE_API_KEY=your-key go run ./examples/comparison/
```

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

Benchmarks on AMD Ryzen 7 5700G, 512-dimension embeddings, `go test -bench -benchtime=2s`.

**Indexing** (`graph.Build`) — runs once, offline:

| Chunks | Time | Note |
|--------|------|------|
| 100 | 22ms | |
| 500 | 337ms | |
| 1,000 | 1.06s | |
| 2,000 | 2.96s | |
| 5,000 | **10.8s** | brute-force would take ~12.6s here |
| 50,000 | ~3min (est.) | brute-force would take ~20+ hours |

> Indexing uses HNSW (O(n log n)). The constant-factor overhead means HNSW is slightly slower than brute-force below ~5,000 chunks, but removes the hard wall that made large corpora impractical. Indexing happens once; the cost is amortized.

**Retrieval** (`Search`) — runs on every query:

| Indexed chunks | Latency |
|----------------|---------|
| 100 | 86µs |
| 500 | 316µs |
| 1,000 | 432µs |
| 5,000 | **786µs** |

**Persistence:**

| Operation | Time |
|-----------|------|
| Save (100 chunks) | ~2ms |
| Load (100 chunks) | ~23ms (includes HNSW rebuild) |

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
