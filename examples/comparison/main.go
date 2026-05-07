// Comparison between classic RAG and e-rag using the project README as corpus.
//
// Both approaches use the same chunks and embeddings for a fair comparison.
//
// Usage:
//
//	VOYAGE_API_KEY=your-key go run ./examples/comparison/
package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/config"
	"github.com/mpsiqueira/e-rag/internal/embedding"
	"github.com/mpsiqueira/e-rag/internal/retriever"
)

var queries = []string{
	"how does the retriever component work?",
	"what are the exact match scores on open-domain QA datasets?",
	"what is the difference between RAG-Sequence and RAG-Token?",
	"how does index hot-swapping work without retraining?",
}

func main() {
	config.Load(".env")

	apiKey := os.Getenv("VOYAGE_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "error: VOYAGE_API_KEY not set")
		os.Exit(1)
	}

	corpusPath := "examples/comparison/rag_paper.txt"
	if _, err := os.Stat(corpusPath); os.IsNotExist(err) {
		corpusPath = "rag_paper.txt"
	}

	text, err := os.ReadFile(corpusPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading corpus: %v\n", err)
		os.Exit(1)
	}

	client := embedding.New(apiKey)

	fmt.Println("📄 corpus: Lewis et al. 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks")
	fmt.Println("⏳ chunking and embedding (one API call)...")
	fmt.Println()

	// semantic chunking — shared between both approaches
	c := chunker.New()
	chunks, err := c.Chunk(string(text), client.Embed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "chunking error: %v\n", err)
		os.Exit(1)
	}

	// embed all chunks + all queries in a single batch
	texts := make([]string, len(chunks)+len(queries))
	for i, ch := range chunks {
		texts[i] = ch.Text
	}
	for i, q := range queries {
		texts[len(chunks)+i] = q
	}

	allEmbeddings, err := client.Embed(texts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embedding error: %v\n", err)
		os.Exit(1)
	}

	chunkEmbeddings := allEmbeddings[:len(chunks)]
	queryEmbeddings := allEmbeddings[len(chunks):]

	fmt.Printf("✅ %d chunks indexed (%d dimensions)\n\n", len(chunks), len(chunkEmbeddings[0]))
	fmt.Println(strings.Repeat("─", 80))

	// build e-rag index from pre-computed embeddings — zero extra API calls
	cachedEmbedder := makeCached(texts, allEmbeddings)
	k := 5
	if k >= len(chunks) {
		k = len(chunks) - 1
	}
	idx, err := retriever.Build(chunks, chunkEmbeddings, k, cachedEmbedder)
	if err != nil {
		fmt.Fprintf(os.Stderr, "index error: %v\n", err)
		os.Exit(1)
	}

	// run comparison for each query
	for i, query := range queries {
		fmt.Printf("\n🔍 query: \"%s\"\n", query)
		fmt.Println(strings.Repeat("─", 80))

		// Classic RAG: top-3 by cosine similarity, no graph awareness
		fmt.Println("\n📦 CLASSIC RAG  (top-3 by cosine similarity)")
		classicResults := classicSearch(queryEmbeddings[i], chunkEmbeddings, chunks, 3)
		for j, r := range classicResults {
			fmt.Printf("  [%d] score=%.4f\n      %s\n\n", j+1, r.score, preview(r.text, 120))
		}

		// e-rag: anchor + semantic community
		fmt.Println("🧠 E-RAG  (anchor + semantic community)")
		result, err := idx.SearchByVector(queryEmbeddings[i], 3)
		if err != nil || result == nil {
			fmt.Println("  no result")
			continue
		}

		fmt.Printf("  anchor  score=%.4f  community=%d\n      %s\n",
			result.Anchor.Score, result.Anchor.CommunityID, preview(result.Anchor.Text, 120))

		if len(result.Community) > 0 {
			fmt.Println("  community neighbors (same semantic region):")
			for _, c := range result.Community {
				fmt.Printf("    · score=%.4f  %s\n", c.Score, preview(c.Text, 100))
			}
		} else {
			fmt.Println("  (anchor is the only member of its community)")
		}

		fmt.Println()
		fmt.Println(strings.Repeat("─", 80))
	}
}

type classicResult struct {
	text  string
	score float32
}

// classicSearch implements naive RAG: top-k chunks by cosine similarity only.
func classicSearch(query []float32, embeddings [][]float32, chunks []chunker.Chunk, k int) []classicResult {
	type scored struct {
		idx   int
		score float32
	}

	results := make([]scored, len(embeddings))
	for i, emb := range embeddings {
		results[i] = scored{i, cosine(query, emb)}
	}
	sort.Slice(results, func(a, b int) bool {
		return results[a].score > results[b].score
	})

	if k > len(results) {
		k = len(results)
	}
	out := make([]classicResult, k)
	for i := range k {
		out[i] = classicResult{text: chunks[results[i].idx].Text, score: results[i].score}
	}
	return out
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(na))*math.Sqrt(float64(nb)))
}

func preview(s string, max int) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), "\n", " ")
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func makeCached(texts []string, embeddings [][]float32) func([]string) ([][]float32, error) {
	cache := make(map[string][]float32, len(texts))
	for i, t := range texts {
		cache[t] = embeddings[i]
	}
	return func(inputs []string) ([][]float32, error) {
		result := make([][]float32, len(inputs))
		for i, t := range inputs {
			result[i] = cache[t]
		}
		return result, nil
	}
}
