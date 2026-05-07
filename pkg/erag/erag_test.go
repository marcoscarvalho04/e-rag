package erag_test

import (
	"path/filepath"
	"testing"

	"github.com/mpsiqueira/e-rag/pkg/erag"
)

// embedder fake que separa economia de clima por palavras-chave.
func fakeEmbedder(texts []string) ([][]float32, error) {
	economia := []string{"juros", "banco", "crédito", "inflação", "taxa"}
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

var testDocs = []string{
	"O banco central elevou a taxa de juros. O crédito ficou mais caro. A inflação desacelerou.",
	"A chuva voltou após semanas de seca. A temperatura caiu bastante. O vento derrubou árvores.",
}

func TestNew_BuildsIndex(t *testing.T) {
	idx, err := erag.New(testDocs, fakeEmbedder)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if idx == nil {
		t.Fatal("esperava índice, got nil")
	}
}

func TestSearch_ReturnsRelevantAnchor(t *testing.T) {
	idx, _ := erag.New(testDocs, fakeEmbedder)

	result, err := idx.Search("impacto dos juros no banco", 3)
	if err != nil {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("esperava resultado")
	}

	if !containsAny(result.Anchor.Text, []string{"juros", "banco", "crédito", "inflação"}) {
		t.Errorf("âncora deveria ser sobre economia, got: %q", result.Anchor.Text)
	}
}

func TestSearch_ScoresBetweenZeroAndOne(t *testing.T) {
	idx, _ := erag.New(testDocs, fakeEmbedder)
	result, _ := idx.Search("juros e crédito", 5)

	if result.Anchor.Score < 0 || result.Anchor.Score > 1 {
		t.Errorf("score âncora fora do range [0,1]: %f", result.Anchor.Score)
	}
	for _, c := range result.Community {
		if c.Score < 0 || c.Score > 1 {
			t.Errorf("score community fora do range [0,1]: %f", c.Score)
		}
	}
}

func TestSearch_TopKRespected(t *testing.T) {
	idx, _ := erag.New(testDocs, fakeEmbedder)
	result, _ := idx.Search("qualquer query", 1)

	if len(result.Community) > 1 {
		t.Errorf("topK=1: esperava no máximo 1 em Community, got %d", len(result.Community))
	}
}

func TestSaveLoad_RoundTrip(t *testing.T) {
	idx, _ := erag.New(testDocs, fakeEmbedder)
	path := filepath.Join(t.TempDir(), "test.gob")

	if err := idx.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := erag.Load(path, fakeEmbedder)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	r1, _ := idx.Search("juros banco", 3)
	r2, _ := loaded.Search("juros banco", 3)

	if r1.Anchor.Text != r2.Anchor.Text {
		t.Errorf("âncora diverge após round-trip:\n  original: %q\n  loaded:   %q",
			r1.Anchor.Text, r2.Anchor.Text)
	}
}

func TestWithOptions(t *testing.T) {
	idx, err := erag.New(testDocs, fakeEmbedder,
		erag.WithK(2),
		erag.WithWindowSize(2),
		erag.WithBreakThreshold(0.3),
	)
	if err != nil {
		t.Fatalf("New com options: %v", err)
	}
	if idx == nil {
		t.Fatal("esperava índice com options customizadas")
	}
}

func TestNew_EmptyDocuments(t *testing.T) {
	_, err := erag.New([]string{}, fakeEmbedder)
	if err == nil {
		t.Error("esperava erro com documentos vazios")
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
