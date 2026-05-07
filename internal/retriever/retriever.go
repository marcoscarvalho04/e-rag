package retriever

import (
	"math"
	"sort"

	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/community"
	"github.com/mpsiqueira/e-rag/internal/graph"
)

// ChunkResult é um chunk retornado com seu score de relevância.
type ChunkResult struct {
	Text        string
	Score       float32
	NodeID      uint32
	CommunityID int
}

// SearchResult contém o chunk âncora e os vizinhos semânticos da sua comunidade.
type SearchResult struct {
	Anchor    ChunkResult
	Community []ChunkResult
}

// Index mantém o estado completo do pipeline indexado.
type Index struct {
	chunks    []chunker.Chunk
	graph     *graph.Graph
	community *community.Result
	embedder  func([]string) ([][]float32, error)
}

// Build constrói o índice a partir de chunks e embeddings já computados.
func Build(
	chunks []chunker.Chunk,
	embeddings [][]float32,
	k int,
	embedder func([]string) ([][]float32, error),
) (*Index, error) {
	g, err := graph.Build(embeddings, k)
	if err != nil {
		return nil, err
	}

	comm := community.Detect(g)

	return &Index{
		chunks:    chunks,
		graph:     g,
		community: comm,
		embedder:  embedder,
	}, nil
}

// Search recebe uma query em texto e retorna o chunk âncora
// e os vizinhos semânticos da sua comunidade, rankeados por relevância.
func (idx *Index) Search(query string, topK int) (*SearchResult, error) {
	embeddings, err := idx.embedder([]string{query})
	if err != nil {
		return nil, err
	}
	return idx.SearchByVector(embeddings[0], topK)
}

// SearchByVector executa o retrieval com um embedding já computado.
// Útil para evitar chamadas extras à API quando o embedding da query já existe.
func (idx *Index) SearchByVector(queryVec []float32, topK int) (*SearchResult, error) {
	// encontra o chunk mais próximo da query no grafo
	nearest := idx.graph.Search(queryVec, 1)
	if len(nearest) == 0 {
		return nil, nil
	}

	anchorID := nearest[0].To
	anchorScore := nearest[0].Weight
	communityID := idx.community.NodeCommunity[anchorID]

	anchor := ChunkResult{
		Text:        idx.chunks[anchorID].Text,
		Score:       anchorScore,
		NodeID:      anchorID,
		CommunityID: communityID,
	}

	// rankeia membros da comunidade por similaridade com a query
	members := idx.community.Members[communityID]
	ranked := make([]ChunkResult, 0, len(members))

	for _, nodeID := range members {
		if nodeID == anchorID {
			continue
		}
		nodeEmb := idx.graph.Nodes[nodeID].Embedding
		score := cosineSimilarity(queryVec, nodeEmb)
		ranked = append(ranked, ChunkResult{
			Text:        idx.chunks[nodeID].Text,
			Score:       score,
			NodeID:      nodeID,
			CommunityID: communityID,
		})
	}

	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].Score > ranked[j].Score
	})

	if topK > 0 && len(ranked) > topK {
		ranked = ranked[:topK]
	}

	return &SearchResult{
		Anchor:    anchor,
		Community: ranked,
	}, nil
}

func cosineSimilarity(a []float32, b []float32) float32 {
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
