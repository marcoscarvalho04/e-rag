package bench

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/mpsiqueira/e-rag/pkg/erag"
)

// syntheticEmbedder gera vetores aleatórios reproduzíveis.
// Dimensão 512 replica o voyage-3-lite.
func syntheticEmbedder(dim int) erag.Embedder {
	rng := rand.New(rand.NewSource(42))
	return func(texts []string) ([][]float32, error) {
		result := make([][]float32, len(texts))
		for i := range texts {
			vec := make([]float32, dim)
			var norm float32
			for j := range vec {
				v := float32(rng.NormFloat64())
				vec[j] = v
				norm += v * v
			}
			// normaliza para unit vector — igual ao que modelos reais produzem
			norm = float32(1.0 / float64(norm))
			for j := range vec {
				vec[j] *= norm
			}
			result[i] = vec
		}
		return result, nil
	}
}

// syntheticDocs gera n documentos com s sentenças cada.
func syntheticDocs(n, sentencesPerDoc int) []string {
	docs := make([]string, n)
	for i := range docs {
		doc := ""
		for j := range sentencesPerDoc {
			doc += fmt.Sprintf("Esta é a sentença %d do documento %d, com conteúdo variado para simular texto real. ", j+1, i+1)
		}
		docs[i] = doc
	}
	return docs
}

// --- Benchmarks de indexação (New) ---

func BenchmarkNew_10docs(b *testing.B) {
	benchmarkNew(b, 10, 5)
}

func BenchmarkNew_50docs(b *testing.B) {
	benchmarkNew(b, 50, 5)
}

func BenchmarkNew_100docs(b *testing.B) {
	benchmarkNew(b, 100, 5)
}

func BenchmarkNew_500docs(b *testing.B) {
	benchmarkNew(b, 500, 3)
}

func benchmarkNew(b *testing.B, nDocs, sentencesPerDoc int) {
	b.Helper()
	embedder := syntheticEmbedder(512)
	docs := syntheticDocs(nDocs, sentencesPerDoc)

	b.ResetTimer()
	for range b.N {
		idx, err := erag.New(docs, embedder, erag.WithK(5))
		if err != nil {
			b.Fatal(err)
		}
		_ = idx
	}
}

// --- Benchmarks de retrieval (Search) ---

func BenchmarkSearch_10docs(b *testing.B) {
	benchmarkSearch(b, 10, 5)
}

func BenchmarkSearch_50docs(b *testing.B) {
	benchmarkSearch(b, 50, 5)
}

func BenchmarkSearch_100docs(b *testing.B) {
	benchmarkSearch(b, 100, 5)
}

func BenchmarkSearch_500docs(b *testing.B) {
	benchmarkSearch(b, 500, 3)
}

func benchmarkSearch(b *testing.B, nDocs, sentencesPerDoc int) {
	b.Helper()
	embedder := syntheticEmbedder(512)
	docs := syntheticDocs(nDocs, sentencesPerDoc)

	idx, err := erag.New(docs, embedder, erag.WithK(5))
	if err != nil {
		b.Fatal(err)
	}

	query := "sentença de busca representativa para medir latência de retrieval"

	b.ResetTimer()
	for range b.N {
		result, err := idx.Search(query, 5)
		if err != nil {
			b.Fatal(err)
		}
		_ = result
	}
}

// --- Benchmarks de persistência (Save / Load) ---

func BenchmarkSave_100docs(b *testing.B) {
	embedder := syntheticEmbedder(512)
	idx, _ := erag.New(syntheticDocs(100, 5), embedder)
	path := filepath.Join(b.TempDir(), "index.gob")

	b.ResetTimer()
	for range b.N {
		if err := idx.Save(path); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoad_100docs(b *testing.B) {
	embedder := syntheticEmbedder(512)
	idx, _ := erag.New(syntheticDocs(100, 5), embedder)
	path := filepath.Join(b.TempDir(), "index.gob")
	idx.Save(path)

	b.ResetTimer()
	for range b.N {
		loaded, err := erag.Load(path, embedder)
		if err != nil {
			b.Fatal(err)
		}
		_ = loaded
	}
}

// --- Benchmark de scaling: mede como graph.Build cresce com n ---

func BenchmarkScaling(b *testing.B) {
	sizes := []int{10, 25, 50, 100, 200}
	embedder := syntheticEmbedder(512)

	for _, n := range sizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			docs := syntheticDocs(n, 3)
			b.ResetTimer()
			for range b.N {
				idx, err := erag.New(docs, embedder, erag.WithK(5))
				if err != nil {
					b.Fatal(err)
				}
				_ = idx
			}
		})
	}
}
