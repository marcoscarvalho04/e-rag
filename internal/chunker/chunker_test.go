package chunker

import (
	"math"
	"testing"
)

// embedFake simula embeddings onde sentenças do mesmo assunto
// têm vetores próximos e sentenças de assuntos diferentes têm vetores distantes.
func embedFake(sentences []string) ([][]float32, error) {
	// dois assuntos simulados: economia (ângulo ~0) e clima (ângulo ~90°)
	economia := []string{"juros", "banco", "crédito", "inflação", "taxa"}
	clima := []string{"chuva", "temperatura", "clima", "vento", "seco"}

	result := make([][]float32, len(sentences))
	for i, s := range sentences {
		s = lowercase(s)
		switch {
		case containsAny(s, economia):
			result[i] = []float32{1.0, 0.0}
		case containsAny(s, clima):
			result[i] = []float32{0.0, 1.0}
		default:
			result[i] = []float32{0.5, 0.5}
		}
	}
	return result, nil
}

func TestChunk_DetectsBoundary(t *testing.T) {
	c := New()

	text := `O banco central elevou a taxa de juros. O crédito ficou mais caro.
	A inflação desacelerou com a medida. A chuva voltou após semanas de seca.
	A temperatura caiu bastante na região. O vento forte derrubou árvores.`

	chunks, err := c.Chunk(text, embedFake)
	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) < 2 {
		t.Errorf("esperava ao menos 2 chunks (economia e clima), got %d", len(chunks))
	}
}

func TestChunk_SingleSentence(t *testing.T) {
	c := New()
	text := "O banco central elevou a taxa de juros."

	chunks, err := c.Chunk(text, embedFake)
	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) != 1 {
		t.Errorf("esperava 1 chunk, got %d", len(chunks))
	}
}

func TestChunk_HomogeneousText(t *testing.T) {
	c := New()

	// texto homogêneo não deve ser fragmentado
	text := `O banco central elevou a taxa de juros. O crédito ficou mais caro.
	A inflação desacelerou com a medida. Os juros impactam o consumo.`

	chunks, err := c.Chunk(text, embedFake)
	if err != nil {
		t.Fatal(err)
	}

	if len(chunks) != 1 {
		t.Errorf("texto homogêneo deveria gerar 1 chunk, got %d", len(chunks))
	}
}

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float64{1, 0}
	if got := cosineSimilarity(a, a); math.Abs(got-1.0) > 1e-9 {
		t.Errorf("vetores idênticos: esperava 1.0, got %f", got)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{0, 1}
	if got := cosineSimilarity(a, b); math.Abs(got) > 1e-9 {
		t.Errorf("vetores ortogonais: esperava 0.0, got %f", got)
	}
}

func TestSplitSentences(t *testing.T) {
	text := "Primeira sentença. Segunda sentença! Terceira sentença?"
	got := splitSentences(text)
	if len(got) != 3 {
		t.Errorf("esperava 3 sentenças, got %d: %v", len(got), got)
	}
}

// helpers do teste
func lowercase(s string) string {
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 32
		}
		result[i] = c
	}
	return string(result)
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
