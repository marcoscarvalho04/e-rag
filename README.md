# e-rag

Enhanced RAG — retrieval augmented generation with semantic context preservation.

Classic RAG treats chunks as isolated vectors. e-rag builds a semantic affinity graph between chunks and uses community detection to return coherent regions of knowledge — not isolated fragments.

## The problem with classic RAG

```
document: A → B → C → D  (sentences with dependencies)
classic:  {A}, {B}, {C}, {D}  (independent chunks)
retrieval: returns C  →  but C without B makes no sense
```

When you cut text into fixed-size chunks, you destroy the context that gives each piece meaning. The chunk is a brick, and the mortar — the semantic context — is lost.

## How e-rag solves it

**Semantic chunking** — instead of fixed sizes, boundaries are detected by measuring semantic drift between sentences via a sliding window. Chunks are cut where ideas actually change.

**K-NN graph** — after embedding, each chunk connects to its K most semantically similar neighbors. The graph makes explicit what classic RAG leaves implicit.

**Louvain community detection** — the graph's natural communities emerge without any arbitrary threshold. The algorithm finds the structure that already exists in the data.

**Community-aware retrieval** — a query returns not just the nearest chunk, but the coherent semantic neighborhood it belongs to. The LLM receives context, not fragments.

```
query → nearest chunk (anchor) → its community → ranked by relevance
```

## Install

```bash
go get github.com/mpsiqueira/e-rag
```

Requires Go 1.22+.

## Usage

```go
import "github.com/mpsiqueira/e-rag/pkg/erag"

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

### Persist and load

```go
// save index to disk — embedding calls happen once, at index time
if err := idx.Save("knowledge.gob"); err != nil {
    log.Fatal(err)
}

// load and serve — no API calls on startup
idx, err := erag.Load("knowledge.gob", embedder)
```

### Options

```go
idx, err := erag.New(docs, embedder,
    erag.WithK(10),                  // K-NN neighbors per chunk (default: 5)
    erag.WithWindowSize(4),          // sliding window size (default: 3)
    erag.WithBreakThreshold(0.3),    // boundary sensitivity (default: 0.25)
)
```

`K` controls how many neighbors each chunk connects to in the graph. Higher values capture more relationships but increase indexing cost.

`WindowSize` controls how many sentences on each side of a candidate boundary are averaged before computing similarity. Larger windows smooth out noise.

`BreakThreshold` controls how sharp a semantic drop must be to become a chunk boundary. Higher values produce smaller, more granular chunks.

## Embedder examples

e-rag is embedder-agnostic. Any function with the signature `func([]string) ([][]float32, error)` works.

**Voyage AI**

```go
func voyageEmbedder(texts []string) ([][]float32, error) {
    // https://www.voyageai.com
    // model: voyage-3-lite
}
```

**OpenAI**

```go
func openAIEmbedder(texts []string) ([][]float32, error) {
    // model: text-embedding-3-small
}
```

**Ollama (local)**

```go
func ollamaEmbedder(texts []string) ([][]float32, error) {
    // model: nomic-embed-text
    // zero API cost, runs locally
}
```

## CLI tools

The repository includes two CLI tools as usage examples.

**Index a document**

```bash
go run ./cmd/erag-index/ -input document.txt -output index.gob -k 5
```

**Serve over HTTP**

```bash
VOYAGE_API_KEY=your-key go run ./cmd/erag-serve/

# index documents
curl -X POST http://localhost:8080/indexes/myindex \
  -H "Content-Type: application/json" \
  -d '{"documents": ["text one...", "text two..."], "k": 5}'

# search
curl -X POST http://localhost:8080/indexes/myindex/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "top_k": 5}'
```

Available endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/indexes` | List all indexes |
| `POST` | `/indexes/{name}` | Create or update index |
| `DELETE` | `/indexes/{name}` | Delete index |
| `POST` | `/indexes/{name}/search` | Search index |

The HTTP server is provided as a reference implementation. For production use, wrap the `pkg/erag` library directly in your application.

## Performance

Benchmarks measured on AMD Ryzen 7 5700G, 512-dimension embeddings, `go test -bench -benchtime=3s`.

**Indexing** (`New`) — runs once, offline:

| Documents | Time |
|-----------|------|
| 10 | 0.5ms |
| 50 | 5ms |
| 100 | 15ms |
| 500 | 126ms |

Indexing is O(n²) due to K-NN graph construction. For corpora above ~10,000 chunks, consider pre-building the index offline. The graph construction is the target for future HNSW optimization.

**Retrieval** (`Search`) — runs on every query:

| Indexed chunks | Latency |
|----------------|---------|
| 10 | 10µs |
| 50 | 36µs |
| 100 | 72µs |
| 500 | 242µs |

Retrieval is linear with index size. At 100 chunks, **72µs per query** — fast enough that no result caching is needed for typical workloads.

**Persistence:**

| Operation | Time |
|-----------|------|
| Save (100 chunks) | ~2ms |
| Load (100 chunks) | ~1.3ms |

Cold start is under 2ms regardless of index size in typical corpora.

## How it compares

| | Classic RAG | GraphRAG (Microsoft) | e-rag |
|---|---|---|---|
| Chunk boundary | Fixed size | Fixed size | Semantic (sliding window) |
| Context preservation | None | LLM-generated metadata | Graph structure |
| LLM calls at index time | None | O(n) — expensive | None |
| Retrieval unit | Isolated chunk | Community | Community |
| Cost | Low | High | Low |

GraphRAG uses an LLM to generate context descriptions for each chunk at indexing time. With n chunks, that means n LLM calls — which is expensive and slow. e-rag derives the same community structure mathematically from the embeddings themselves.

## Architecture

```
documents
    ↓
semantic chunking       sliding window over sentences detects topic shifts
    ↓
embeddings              any model, one batch call per document set
    ↓
K-NN graph              each chunk connects to K nearest neighbors by cosine similarity
    ↓
Louvain communities     natural clusters emerge without any arbitrary threshold
    ↓
persisted index         chunks + graph + communities serialized to disk

query
    ↓
embed query
    ↓
graph search            finds anchor chunk via cosine similarity
    ↓
community retrieval     returns anchor + semantic neighbors, ranked by relevance
```

## License

MIT
