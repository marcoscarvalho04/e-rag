package graph

import (
	"math"
	"sort"

	"github.com/ylerby/vecgo"
	hnswpkg "github.com/ylerby/vecgo/index/hnsw"
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

	// idx é o índice HNSW usado para busca rápida em query-time.
	// Não é serializado — reconstruído em Build e em Load via RebuildIndex.
	idx *vecgo.Vecgo[uint32]
}

// Build constrói o grafo K-NN a partir dos embeddings usando HNSW.
//
// Complexidade: O(n log n) — substituiu a abordagem anterior O(n²).
// O HNSW é usado tanto para extrair os K vizinhos de cada nó durante a
// indexação quanto para buscas rápidas em query-time.
func Build(embeddings [][]float32, k int) (*Graph, error) {
	n := len(embeddings)

	idx := newHNSW()
	norms := make([][]float32, n)
	for i, emb := range embeddings {
		norms[i] = normalize(emb)
		if _, err := idx.Insert(vecgo.VectorWithData[uint32]{
			Vector: norms[i],
			Data:   uint32(i),
		}); err != nil {
			return nil, err
		}
	}

	nodes := make([]Node, n)
	edges := make([][]Edge, n)

	for i, emb := range embeddings {
		nodes[i] = Node{ID: uint32(i), Embedding: emb}

		// k+1 porque o próprio nó pode aparecer nos resultados
		results, err := idx.KNNSearch(norms[i], k+1)
		if err != nil {
			return nil, err
		}

		neighbors := make([]Edge, 0, k)
		for _, r := range results {
			if r.Data == uint32(i) {
				continue
			}
			if len(neighbors) >= k {
				break
			}
			w := cosineSimilarity(emb, embeddings[r.Data])
			neighbors = append(neighbors, Edge{To: r.Data, Weight: w})
		}
		edges[i] = neighbors
	}

	return &Graph{Nodes: nodes, Edges: edges, idx: idx}, nil
}

// RebuildIndex reconstrói o índice HNSW a partir dos Nodes existentes.
// Deve ser chamado após deserialização (Load) para restaurar buscas O(log n).
func (g *Graph) RebuildIndex() error {
	idx := newHNSW()
	for _, n := range g.Nodes {
		if _, err := idx.Insert(vecgo.VectorWithData[uint32]{
			Vector: normalize(n.Embedding),
			Data:   n.ID,
		}); err != nil {
			return err
		}
	}
	g.idx = idx
	return nil
}

// Search encontra os K nós mais próximos de um embedding de query via HNSW.
// Se o índice não estiver disponível, faz busca exaustiva como fallback.
func (g *Graph) Search(query []float32, k int) []Edge {
	if g.idx != nil {
		results, err := g.idx.KNNSearch(normalize(query), k)
		if err == nil {
			edges := make([]Edge, 0, len(results))
			for _, r := range results {
				edges = append(edges, Edge{
					To:     r.Data,
					Weight: cosineSimilarity(query, g.Nodes[r.Data].Embedding),
				})
			}
			return edges
		}
	}
	return g.bruteSearch(query, k)
}

// Neighbors retorna os vizinhos de um nó pelo seu ID.
func (g *Graph) Neighbors(id uint32) []Edge {
	return g.Edges[id]
}

// bruteSearch é o fallback O(n) usado quando o índice HNSW não está disponível.
func (g *Graph) bruteSearch(query []float32, k int) []Edge {
	type scored struct {
		id     uint32
		weight float32
	}
	candidates := make([]scored, len(g.Nodes))
	for i, n := range g.Nodes {
		candidates[i] = scored{n.ID, cosineSimilarity(query, n.Embedding)}
	}
	sort.Slice(candidates, func(a, b int) bool {
		return candidates[a].weight > candidates[b].weight
	})
	if k > len(candidates) {
		k = len(candidates)
	}
	results := make([]Edge, k)
	for i, c := range candidates[:k] {
		results[i] = Edge{To: c.id, Weight: c.weight}
	}
	return results
}

func newHNSW() *vecgo.Vecgo[uint32] {
	return vecgo.NewHNSW[uint32](func(o *hnswpkg.Options) {
		o.M = 16
		o.EF = 100
	})
}

func normalize(v []float32) []float32 {
	var norm float32
	for _, x := range v {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm == 0 {
		return v
	}
	result := make([]float32, len(v))
	for i, x := range v {
		result[i] = x / norm
	}
	return result
}

func cosineSimilarity(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}
