// Showcase: classic RAG vs e-rag on a heterogeneous corpus.
//
// Three completely different domains are indexed together:
//   - ML: the original RAG paper (Lewis et al., 2020)
//   - Sports: sports periodization and athletic training
//   - Finance: index funds and portfolio management
//
// Queries are designed to be lexically ambiguous across domains.
// Classic RAG returns top-k fragments that may span multiple domains.
// E-rag returns a coherent semantic community from the correct domain.
// Both contexts are sent to Claude so the difference in answer quality is visible.
//
// Usage:
//
//	VOYAGE_API_KEY=your-key ANTHROPIC_API_KEY=your-key go run ./examples/showcase/
package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/config"
	"github.com/mpsiqueira/e-rag/internal/embedding"
	"github.com/mpsiqueira/e-rag/internal/retriever"
)

type corpus struct {
	label string
	file  string
}

var corpora = []corpus{
	{"ML / RAG Paper", "rag_paper.txt"},
	{"Sports / Periodization", "sports_training.txt"},
	{"Finance / Index Funds", "finance_index.txt"},
	{"Climate Science", "climate_models.txt"},
	{"Medicine / Clinical Trials", "clinical_trials.txt"},
	{"Software Architecture", "software_architecture.txt"},
}

// Queries are maximally ambiguous across all 6 domains.
// Every keyword ("model", "parameters", "performance", "training",
// "index", "retrieval", "adaptation") appears in multiple domains
// with completely different meanings.
var queries = []string{
	"how does the model adapt its parameters over time?",
	"what training approach leads to the best long-term performance?",
	"how is the index structured for efficient retrieval?",
}

func main() {
	config.Load(".env")

	voyageKey := os.Getenv("VOYAGE_API_KEY")
	if voyageKey == "" {
		fmt.Fprintln(os.Stderr, "error: VOYAGE_API_KEY not set")
		os.Exit(1)
	}
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "error: ANTHROPIC_API_KEY not set")
		os.Exit(1)
	}

	corpusDir := resolveDir(
		"examples/showcase/corpora",
		"corpora",
	)

	embClient := embedding.New(voyageKey)
	llm := anthropic.NewClient()

	// chunk each corpus separately so we can track which domain each chunk came from
	c := chunker.New()
	var allChunks []chunker.Chunk
	var chunkLabel []string // parallel slice: domain label per chunk

	fmt.Println("⏳ semantic chunking across 3 domains...")
	for _, corp := range corpora {
		text, err := os.ReadFile(filepath.Join(corpusDir, corp.file))
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading %s: %v\n", corp.file, err)
			os.Exit(1)
		}
		chunks, err := c.Chunk(string(text), embClient.Embed)
		if err != nil {
			fmt.Fprintf(os.Stderr, "chunking %s: %v\n", corp.file, err)
			os.Exit(1)
		}
		for range chunks {
			chunkLabel = append(chunkLabel, corp.label)
		}
		allChunks = append(allChunks, chunks...)
		fmt.Printf("  %-28s → %d chunks\n", corp.label, len(chunks))
	}

	// embed all chunks + all queries in a single batch
	fmt.Println("⏳ embedding all chunks + queries (one batch)...")
	texts := make([]string, len(allChunks)+len(queries))
	for i, ch := range allChunks {
		texts[i] = ch.Text
	}
	for i, q := range queries {
		texts[len(allChunks)+i] = q
	}

	allEmbs, err := embClient.Embed(texts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embedding error: %v\n", err)
		os.Exit(1)
	}

	chunkEmbs := allEmbs[:len(allChunks)]
	queryEmbs := allEmbs[len(allChunks):]

	// label index: text → domain label (for display)
	labelByText := make(map[string]string, len(allChunks))
	for i, ch := range allChunks {
		labelByText[ch.Text] = chunkLabel[i]
	}

	fmt.Printf("\n✅ %d total chunks indexed across 3 domains (%d dims)\n", len(allChunks), len(chunkEmbs[0]))
	fmt.Println(strings.Repeat("═", 80))

	// build e-rag index from pre-computed embeddings
	cached := makeCached(texts, allEmbs)
	k := 5
	if k >= len(allChunks) {
		k = len(allChunks) - 1
	}
	idx, err := retriever.Build(allChunks, chunkEmbs, k, cached)
	if err != nil {
		fmt.Fprintf(os.Stderr, "index error: %v\n", err)
		os.Exit(1)
	}

	for qi, query := range queries {
		fmt.Printf("\n🔍 QUERY: \"%s\"\n", query)
		fmt.Println(strings.Repeat("─", 80))

		// ── Classic RAG ──────────────────────────────────────────────────────
		classicHits := classicSearch(queryEmbs[qi], chunkEmbs, allChunks, 3)

		fmt.Println("\n📦 CLASSIC RAG — top-3 by cosine similarity")
		for j, r := range classicHits {
			domain := labelByText[r.text]
			fmt.Printf("  [%d] %-28s score=%.4f\n      %s\n\n",
				j+1, tag(domain), r.score, preview(r.text, 110))
		}

		// ── E-RAG ─────────────────────────────────────────────────────────────
		eragResult, err := idx.SearchByVector(queryEmbs[qi], 3)
		fmt.Println("🧠 E-RAG — anchor + semantic community")
		if err != nil || eragResult == nil {
			fmt.Println("  no result")
		} else {
			anchorDomain := labelByText[eragResult.Anchor.Text]
			fmt.Printf("  anchor  %-28s score=%.4f  community=%d\n          %s\n",
				tag(anchorDomain), eragResult.Anchor.Score, eragResult.Anchor.CommunityID,
				preview(eragResult.Anchor.Text, 110))
			if len(eragResult.Community) > 0 {
				fmt.Println("  community:")
				for _, cm := range eragResult.Community {
					d := labelByText[cm.Text]
					fmt.Printf("    · %-28s score=%.4f  %s\n",
						tag(d), cm.Score, preview(cm.Text, 90))
				}
			}
		}

		// ── LLM comparison ───────────────────────────────────────────────────
		fmt.Println()
		classicCtx := buildClassicContext(classicHits)
		eragCtx := buildEragContext(eragResult)

		fmt.Println("┌─ LLM answer — Classic RAG context " + strings.Repeat("─", 44) + "┐")
		classicAnswer := generate(llm, query, classicCtx)
		for _, line := range wrapLines(classicAnswer, 76) {
			fmt.Println("│ " + line)
		}
		fmt.Println("└" + strings.Repeat("─", 78) + "┘")

		fmt.Println()
		fmt.Println("┌─ LLM answer — e-rag context " + strings.Repeat("─", 50) + "┐")
		eragAnswer := generate(llm, query, eragCtx)
		for _, line := range wrapLines(eragAnswer, 76) {
			fmt.Println("│ " + line)
		}
		fmt.Println("└" + strings.Repeat("─", 78) + "┘")

		fmt.Println()
		fmt.Println(strings.Repeat("═", 80))
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func generate(client anthropic.Client, query, ragCtx string) string {
	prompt := fmt.Sprintf(
		"Answer the question using ONLY the context below. Be direct and specific (3-4 sentences).\n\nContext:\n%s\nQuestion: %s",
		ragCtx, query,
	)
	msg, err := client.Messages.New(context.TODO(), anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeHaiku4_5_20251001,
		MaxTokens: 300,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	})
	if err != nil {
		return fmt.Sprintf("(error calling Claude: %v)", err)
	}
	return msg.Content[0].Text
}

func buildClassicContext(hits []classicResult) string {
	var sb strings.Builder
	for i, r := range hits {
		fmt.Fprintf(&sb, "[%d] %s\n\n", i+1, r.text)
	}
	return sb.String()
}

func buildEragContext(r *retriever.SearchResult) string {
	if r == nil {
		return ""
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "[anchor]\n%s\n\n", r.Anchor.Text)
	for i, c := range r.Community {
		fmt.Fprintf(&sb, "[context %d]\n%s\n\n", i+1, c.Text)
	}
	return sb.String()
}

type classicResult struct {
	idx   int
	text  string
	score float32
}

func classicSearch(query []float32, embs [][]float32, chunks []chunker.Chunk, k int) []classicResult {
	type scored struct {
		idx   int
		score float32
	}
	results := make([]scored, len(embs))
	for i, e := range embs {
		results[i] = scored{i, cosine(query, e)}
	}
	sort.Slice(results, func(a, b int) bool { return results[a].score > results[b].score })
	if k > len(results) {
		k = len(results)
	}
	out := make([]classicResult, k)
	for i := range k {
		out[i] = classicResult{idx: results[i].idx, text: chunks[results[i].idx].Text, score: results[i].score}
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

func makeCached(texts []string, embs [][]float32) func([]string) ([][]float32, error) {
	cache := make(map[string][]float32, len(texts))
	for i, t := range texts {
		cache[t] = embs[i]
	}
	return func(inputs []string) ([][]float32, error) {
		out := make([][]float32, len(inputs))
		for i, t := range inputs {
			out[i] = cache[t]
		}
		return out, nil
	}
}

func resolveDir(candidates ...string) string {
	for _, d := range candidates {
		if _, err := os.Stat(d); err == nil {
			return d
		}
	}
	return candidates[len(candidates)-1]
}

func tag(domain string) string {
	return "[" + domain + "]"
}

func preview(s string, max int) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), "\n", " ")
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func wrapLines(s string, width int) []string {
	var lines []string
	for _, paragraph := range strings.Split(s, "\n") {
		if len(paragraph) <= width {
			lines = append(lines, paragraph)
			continue
		}
		words := strings.Fields(paragraph)
		var cur strings.Builder
		for _, w := range words {
			if cur.Len()+1+len(w) > width && cur.Len() > 0 {
				lines = append(lines, cur.String())
				cur.Reset()
			}
			if cur.Len() > 0 {
				cur.WriteByte(' ')
			}
			cur.WriteString(w)
		}
		if cur.Len() > 0 {
			lines = append(lines, cur.String())
		}
	}
	return lines
}
