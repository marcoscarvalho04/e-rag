package retriever

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/community"
	"github.com/mpsiqueira/e-rag/internal/graph"
)

// snapshot é a representação serializável do índice completo.
type snapshot struct {
	Chunks        []chunker.Chunk
	GraphNodes    []graph.Node
	GraphEdges    [][]graph.Edge
	NodeCommunity []int
	Members       map[int][]uint32
}

// Save persiste o índice em disco via gob.
// O embedder não é serializado — deve ser fornecido no Load.
func (idx *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("criando arquivo de índice: %w", err)
	}
	defer f.Close()

	snap := snapshot{
		Chunks:        idx.chunks,
		GraphNodes:    idx.graph.Nodes,
		GraphEdges:    idx.graph.Edges,
		NodeCommunity: idx.community.NodeCommunity,
		Members:       idx.community.Members,
	}

	if err := gob.NewEncoder(f).Encode(snap); err != nil {
		return fmt.Errorf("serializando índice: %w", err)
	}

	return nil
}

// Load restaura um índice a partir de um arquivo salvo por Save.
// O embedder deve ser fornecido — é a única dependência externa do índice.
func Load(path string, embedder func([]string) ([][]float32, error)) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("abrindo arquivo de índice: %w", err)
	}
	defer f.Close()

	var snap snapshot
	if err := gob.NewDecoder(f).Decode(&snap); err != nil {
		return nil, fmt.Errorf("desserializando índice: %w", err)
	}

	g := &graph.Graph{
		Nodes: snap.GraphNodes,
		Edges: snap.GraphEdges,
	}
	if err := g.RebuildIndex(); err != nil {
		return nil, fmt.Errorf("reconstruindo índice HNSW: %w", err)
	}

	comm := &community.Result{
		NodeCommunity: snap.NodeCommunity,
		Members:       snap.Members,
	}

	return &Index{
		chunks:    snap.Chunks,
		graph:     g,
		community: comm,
		embedder:  embedder,
	}, nil
}
