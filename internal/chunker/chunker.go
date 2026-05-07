package chunker

import (
	"math"
	"strings"
	"unicode"
)

type Chunk struct {
	Text     string
	Start    int // índice da primeira sentença
	End      int // índice da última sentença (inclusive)
}

type Chunker struct {
	// tamanho de cada lado da janela deslizante
	WindowSize int
	// quão profundo precisa ser o vale para virar fronteira (0.0 a 1.0)
	BreakThreshold float64
}

func New() *Chunker {
	return &Chunker{
		WindowSize:     3,
		BreakThreshold: 0.25,
	}
}

// Chunk divide o texto em chunks semânticos usando janela deslizante.
// embedFn recebe uma lista de sentenças e retorna seus embeddings na mesma ordem.
func (c *Chunker) Chunk(text string, embedFn func(sentences []string) ([][]float32, error)) ([]Chunk, error) {
	sentences := splitSentences(text)
	if len(sentences) <= 1 {
		return []Chunk{{Text: text, Start: 0, End: len(sentences) - 1}}, nil
	}

	embeddings, err := embedFn(sentences)
	if err != nil {
		return nil, err
	}

	similarities := c.slidingWindowSimilarities(embeddings)
	breakpoints := c.detectBreakpoints(similarities)

	return c.buildChunks(sentences, breakpoints), nil
}

// slidingWindowSimilarities calcula a similaridade entre a janela passada e futura
// para cada posição i no texto.
func (c *Chunker) slidingWindowSimilarities(embeddings [][]float32) []float64 {
	n := len(embeddings)
	similarities := make([]float64, n-1)

	for i := 0; i < n-1; i++ {
		past := windowMean(embeddings, max(0, i-c.WindowSize+1), i+1)
		future := windowMean(embeddings, i+1, min(n, i+1+c.WindowSize))
		similarities[i] = cosineSimilarity(past, future)
	}

	return similarities
}

// detectBreakpoints encontra os mínimos locais na curva de similaridade
// que são suficientemente profundos para indicar mudança de assunto.
func (c *Chunker) detectBreakpoints(similarities []float64) []int {
	if len(similarities) == 0 {
		return nil
	}

	mean, std := stats(similarities)
	threshold := mean - c.BreakThreshold*std

	var breakpoints []int
	for i := 1; i < len(similarities)-1; i++ {
		isLocalMin := similarities[i] < similarities[i-1] && similarities[i] < similarities[i+1]
		isBelowThreshold := similarities[i] < threshold
		if isLocalMin && isBelowThreshold {
			breakpoints = append(breakpoints, i+1) // corta após a sentença i
		}
	}

	return breakpoints
}

func (c *Chunker) buildChunks(sentences []string, breakpoints []int) []Chunk {
	var chunks []Chunk
	start := 0

	for _, bp := range breakpoints {
		chunks = append(chunks, Chunk{
			Text:  strings.Join(sentences[start:bp], " "),
			Start: start,
			End:   bp - 1,
		})
		start = bp
	}

	// último chunk
	chunks = append(chunks, Chunk{
		Text:  strings.Join(sentences[start:], " "),
		Start: start,
		End:   len(sentences) - 1,
	})

	return chunks
}

// splitSentences divide o texto em sentenças por pontuação terminal.
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(strings.TrimSpace(text))
	for i, r := range runes {
		current.WriteRune(r)

		isTerminal := r == '.' || r == '!' || r == '?'
		if !isTerminal {
			continue
		}

		// verifica se o próximo caractere não-espaço é maiúsculo (nova sentença)
		rest := runes[i+1:]
		nextIdx := -1
		for j, nr := range rest {
			if !unicode.IsSpace(nr) {
				nextIdx = j
				break
			}
		}

		isEnd := nextIdx == -1 || unicode.IsUpper(rest[nextIdx])
		if isEnd {
			s := strings.TrimSpace(current.String())
			if s != "" {
				sentences = append(sentences, s)
			}
			current.Reset()
		}
	}

	if remaining := strings.TrimSpace(current.String()); remaining != "" {
		sentences = append(sentences, remaining)
	}

	return sentences
}

// windowMean calcula o vetor médio de um intervalo de embeddings.
func windowMean(embeddings [][]float32, from, to int) []float64 {
	if from >= to {
		return toFloat64(embeddings[from])
	}

	dim := len(embeddings[0])
	mean := make([]float64, dim)

	for i := from; i < to; i++ {
		for j, v := range embeddings[i] {
			mean[j] += float64(v)
		}
	}

	count := float64(to - from)
	for j := range mean {
		mean[j] /= count
	}

	return mean
}

func cosineSimilarity(a, b []float64) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func stats(values []float64) (mean, std float64) {
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	for _, v := range values {
		diff := v - mean
		std += diff * diff
	}
	std = math.Sqrt(std / float64(len(values)))
	return
}

func toFloat64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
