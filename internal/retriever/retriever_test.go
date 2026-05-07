package retriever

import (
	"testing"

	"github.com/mpsiqueira/e-rag/internal/chunker"
)

// embedder fake: retorna embeddings pré-definidos por índice de texto.
// Cluster A: textos sobre economia → vetor (1, 0)
// Cluster B: textos sobre clima    → vetor (0, 1)
func fakeEmbedder(texts []string) ([][]float32, error) {
	economia := []string{"juros", "banco", "crédito", "inflação"}
	result := make([][]float32, len(texts))
	for i, t := range texts {
		if containsAny(t, economia) {
			result[i] = []float32{1.0, 0.0}
		} else {
			result[i] = []float32{0.0, 1.0}
		}
	}
	return result, nil
}

func buildTestIndex(t *testing.T) *Index {
	t.Helper()

	chunks := []chunker.Chunk{
		{Text: "O banco central elevou a taxa de juros.", Start: 0, End: 0},   // A0
		{Text: "O crédito ficou mais caro após a decisão.", Start: 1, End: 1}, // A1
		{Text: "A inflação desacelerou com a medida.", Start: 2, End: 2},      // A2
		{Text: "A chuva voltou após semanas de seca.", Start: 3, End: 3},      // B0
		{Text: "A temperatura caiu bastante na região.", Start: 4, End: 4},    // B1
		{Text: "O vento forte derrubou árvores na cidade.", Start: 5, End: 5}, // B2
	}

	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}

	embeddings, err := fakeEmbedder(texts)
	if err != nil {
		t.Fatal(err)
	}

	idx, err := Build(chunks, embeddings, 2, fakeEmbedder)
	if err != nil {
		t.Fatal(err)
	}

	return idx
}

func TestSearch_AnchorInCorrectCluster(t *testing.T) {
	idx := buildTestIndex(t)

	// query sobre economia deve retornar chunk âncora do cluster A
	result, err := idx.Search("qual o impacto dos juros no banco?", 3)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("esperava resultado, got nil")
	}

	// âncora deve ser do cluster A (índices 0, 1, 2)
	if result.Anchor.NodeID >= 3 {
		t.Errorf("âncora no cluster errado: nodeID=%d (esperava 0-2)", result.Anchor.NodeID)
	}
}

func TestSearch_CommunityInSameCluster(t *testing.T) {
	idx := buildTestIndex(t)

	result, err := idx.Search("qual o impacto dos juros no banco?", 10)
	if err != nil {
		t.Fatal(err)
	}

	// todos os membros da comunidade devem ser do cluster A
	for _, r := range result.Community {
		if r.NodeID >= 3 {
			t.Errorf("comunidade contém chunk do cluster errado: nodeID=%d", r.NodeID)
		}
	}
}

func TestSearch_AnchorNotInCommunity(t *testing.T) {
	idx := buildTestIndex(t)

	result, err := idx.Search("taxa de juros e crédito", 10)
	if err != nil {
		t.Fatal(err)
	}

	// âncora não deve aparecer duplicado em Community
	for _, r := range result.Community {
		if r.NodeID == result.Anchor.NodeID {
			t.Errorf("âncora duplicado em Community: nodeID=%d", r.NodeID)
		}
	}
}

func TestSearch_ScoresDescending(t *testing.T) {
	idx := buildTestIndex(t)

	result, err := idx.Search("juros e inflação", 10)
	if err != nil {
		t.Fatal(err)
	}

	for i := 1; i < len(result.Community); i++ {
		if result.Community[i].Score > result.Community[i-1].Score {
			t.Errorf("scores fora de ordem: [%d]=%f > [%d]=%f",
				i, result.Community[i].Score, i-1, result.Community[i-1].Score)
		}
	}
}

func TestSearch_TopKLimit(t *testing.T) {
	idx := buildTestIndex(t)

	result, err := idx.Search("juros banco crédito", 1)
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Community) > 1 {
		t.Errorf("topK=1: esperava no máximo 1 resultado em Community, got %d", len(result.Community))
	}
}

func containsAny(s string, words []string) bool {
	for _, w := range words {
		if len(s) >= len(w) {
			for i := 0; i <= len(s)-len(w); i++ {
				if s[i:i+len(w)] == w {
					return true
				}
			}
		}
	}
	return false
}
