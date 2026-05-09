package graph

import (
	"math"
	"testing"
)

// embeddings sintéticos: dois clusters bem separados
// cluster A: vetores próximos de (1, 0)
// cluster B: vetores próximos de (0, 1)
func syntheticEmbeddings() [][]float32 {
	return [][]float32{
		{1.00, 0.01}, // A0
		{0.99, 0.02}, // A1
		{0.98, 0.03}, // A2
		{0.01, 1.00}, // B0
		{0.02, 0.99}, // B1
		{0.03, 0.98}, // B2
	}
}

func TestBuild_NeighborsWithinCluster(t *testing.T) {
	embs := syntheticEmbeddings()
	g, err := Build(embs, 2)
	if err != nil {
		t.Fatal(err)
	}

	// nó A0 (índice 0) deve ter vizinhos dentro do cluster A (1, 2)
	neighbors := g.Neighbors(0)
	for _, e := range neighbors {
		if e.To >= 3 {
			t.Errorf("A0 conectou com cluster B (nó %d) — esperava apenas cluster A", e.To)
		}
	}

	// nó B0 (índice 3) deve ter vizinhos dentro do cluster B (4, 5)
	neighbors = g.Neighbors(3)
	for _, e := range neighbors {
		if e.To < 3 {
			t.Errorf("B0 conectou com cluster A (nó %d) — esperava apenas cluster B", e.To)
		}
	}
}

func TestBuild_EdgeWeights(t *testing.T) {
	embs := syntheticEmbeddings()
	g, err := Build(embs, 2)
	if err != nil {
		t.Fatal(err)
	}

	// vizinhos do mesmo cluster devem ter peso alto (>0.99)
	for _, e := range g.Neighbors(0) {
		if e.Weight < 0.99 {
			t.Errorf("peso dentro do cluster deveria ser >0.99, got %f", e.Weight)
		}
	}
}

func TestSearch_FindsNearestCluster(t *testing.T) {
	embs := syntheticEmbeddings()
	g, err := Build(embs, 2)
	if err != nil {
		t.Fatal(err)
	}

	// query no cluster A
	queryA := []float32{1.0, 0.0}
	results := g.Search(queryA, 1)
	if len(results) == 0 {
		t.Fatal("busca não retornou resultados")
	}
	if results[0].To >= 3 {
		t.Errorf("query cluster A deveria retornar nó de A, got nó %d", results[0].To)
	}

	// query no cluster B
	queryB := []float32{0.0, 1.0}
	results = g.Search(queryB, 1)
	if results[0].To < 3 {
		t.Errorf("query cluster B deveria retornar nó de B, got nó %d", results[0].To)
	}
}

func TestBuild_NoSelfLoops(t *testing.T) {
	embs := syntheticEmbeddings()
	g, err := Build(embs, 3)
	if err != nil {
		t.Fatal(err)
	}

	for i, neighbors := range g.Edges {
		for _, e := range neighbors {
			if e.To == uint32(i) {
				t.Errorf("nó %d tem self-loop", i)
			}
		}
	}
}

func TestCosineSimilarity_Sanity(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	if got := cosineSimilarity(a, a); math.Abs(float64(got-1.0)) > 1e-6 {
		t.Errorf("idênticos: esperava 1.0, got %f", got)
	}
	if got := cosineSimilarity(a, b); math.Abs(float64(got)) > 1e-6 {
		t.Errorf("ortogonais: esperava 0.0, got %f", got)
	}
}
