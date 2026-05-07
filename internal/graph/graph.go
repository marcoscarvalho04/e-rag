package graph

import (
	"math"
	"sort"
)

// Node representa um chunk indexado no grafo.
type Node struct {
	ID        uint32
	Embedding []float32
}

// Edge representa uma aresta ponderada entre dois nós.
type Edge struct {
	To     uint32
	Weight float32 // similaridade coseno
}

// Graph é o grafo K-NN esparso construído sobre os embeddings dos chunks.
type Graph struct {
	Nodes []Node
	Edges [][]Edge // lista de adjacência: Edges[i] = vizinhos do nó i
}

// Build constrói o grafo K-NN a partir dos embeddings.
// Para cada nó, conecta os K vizinhos mais próximos por similaridade coseno.
//
// Complexidade: O(n²) — adequado para corpora de até ~10.000 chunks.
// Para corpora maiores, substituir a busca exaustiva por HNSW (Hierarchical
// Navigable Small World), reduzindo para O(n log n) sem alterar a interface.
func Build(embeddings [][]float32, k int) (*Graph, error) {
	n := len(embeddings)
	dims := len(embeddings[0])

	nodes := make([]Node, n)
	edges := make([][]Edge, n)

	for i, emb := range embeddings {
		nodes[i] = Node{ID: uint32(i), Embedding: emb}
	}

	for i := 0; i < n; i++ {
		// calcula similaridade de i com todos os outros nós
		type scored struct {
			id     uint32
			weight float32
		}
		candidates := make([]scored, 0, n-1)

		for j := 0; j < n; j++ {
			if j == i {
				continue
			}
			w := cosineSimilarity(embeddings[i], embeddings[j], dims)
			candidates = append(candidates, scored{uint32(j), w})
		}

		// ordena por similaridade decrescente e pega os K melhores
		sort.Slice(candidates, func(a, b int) bool {
			return candidates[a].weight > candidates[b].weight
		})

		limit := k
		if limit > len(candidates) {
			limit = len(candidates)
		}

		edges[i] = make([]Edge, limit)
		for idx, c := range candidates[:limit] {
			edges[i][idx] = Edge{To: c.id, Weight: c.weight}
		}
	}

	return &Graph{Nodes: nodes, Edges: edges}, nil
}

// Search encontra os K nós mais próximos de um embedding de query.
// Usado no retrieval para encontrar o chunk âncora.
func (g *Graph) Search(query []float32, k int) []Edge {
	dims := len(query)

	type scored struct {
		id     uint32
		weight float32
	}

	candidates := make([]scored, len(g.Nodes))
	for i, n := range g.Nodes {
		candidates[i] = scored{n.ID, cosineSimilarity(query, n.Embedding, dims)}
	}

	sort.Slice(candidates, func(a, b int) bool {
		return candidates[a].weight > candidates[b].weight
	})

	limit := k
	if limit > len(candidates) {
		limit = len(candidates)
	}

	results := make([]Edge, limit)
	for i, c := range candidates[:limit] {
		results[i] = Edge{To: c.id, Weight: c.weight}
	}

	return results
}

// Neighbors retorna os vizinhos de um nó pelo seu ID.
func (g *Graph) Neighbors(id uint32) []Edge {
	return g.Edges[id]
}

func cosineSimilarity(a, b []float32, dims int) float32 {
	var dot, normA, normB float32
	for i := 0; i < dims; i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}
