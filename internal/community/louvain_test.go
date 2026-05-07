package community

import (
	"testing"

	"github.com/mpsiqueira/e-rag/internal/graph"
)

// dois clusters bem separados: A=(1,0), B=(0,1)
func syntheticGraph(t *testing.T) *graph.Graph {
	t.Helper()
	embeddings := [][]float32{
		{1.00, 0.01}, // A0
		{0.99, 0.02}, // A1
		{0.98, 0.03}, // A2
		{0.01, 1.00}, // B0
		{0.02, 0.99}, // B1
		{0.03, 0.98}, // B2
	}
	g, err := graph.Build(embeddings, 2)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestDetect_TwoClusters(t *testing.T) {
	g := syntheticGraph(t)
	result := Detect(g)

	// A0, A1, A2 devem estar na mesma comunidade
	commA := result.NodeCommunity[0]
	if result.NodeCommunity[1] != commA || result.NodeCommunity[2] != commA {
		t.Errorf("A0=%d A1=%d A2=%d — esperava todos na mesma comunidade",
			result.NodeCommunity[0], result.NodeCommunity[1], result.NodeCommunity[2])
	}

	// B0, B1, B2 devem estar na mesma comunidade
	commB := result.NodeCommunity[3]
	if result.NodeCommunity[4] != commB || result.NodeCommunity[5] != commB {
		t.Errorf("B0=%d B1=%d B2=%d — esperava todos na mesma comunidade",
			result.NodeCommunity[3], result.NodeCommunity[4], result.NodeCommunity[5])
	}

	// as duas comunidades devem ser distintas
	if commA == commB {
		t.Error("cluster A e cluster B deveriam estar em comunidades diferentes")
	}
}

func TestDetect_MembersConsistent(t *testing.T) {
	g := syntheticGraph(t)
	result := Detect(g)

	// verifica que Members e NodeCommunity são consistentes
	for commID, members := range result.Members {
		for _, nodeID := range members {
			if result.NodeCommunity[nodeID] != commID {
				t.Errorf("nó %d está em Members[%d] mas NodeCommunity diz %d",
					nodeID, commID, result.NodeCommunity[nodeID])
			}
		}
	}
}

func TestDetect_AllNodesAssigned(t *testing.T) {
	g := syntheticGraph(t)
	result := Detect(g)

	if len(result.NodeCommunity) != len(g.Nodes) {
		t.Errorf("esperava %d nós mapeados, got %d", len(g.Nodes), len(result.NodeCommunity))
	}

	// conta nós em Members — deve bater com o total
	total := 0
	for _, members := range result.Members {
		total += len(members)
	}
	if total != len(g.Nodes) {
		t.Errorf("Members tem %d nós, esperava %d", total, len(g.Nodes))
	}
}

func TestDetect_SingleNode(t *testing.T) {
	embeddings := [][]float32{{1.0, 0.0}}
	g, err := graph.Build(embeddings, 1)
	if err != nil {
		t.Fatal(err)
	}

	result := Detect(g)
	if len(result.Members) != 1 {
		t.Errorf("um único nó deveria formar uma comunidade, got %d", len(result.Members))
	}
}

func TestDetect_CommunityIDsContiguous(t *testing.T) {
	g := syntheticGraph(t)
	result := Detect(g)

	// IDs de comunidade devem ser 0, 1, 2... sem gaps
	for id := range result.Members {
		if id < 0 || id >= len(result.Members) {
			t.Errorf("community ID %d fora do range [0, %d)", id, len(result.Members))
		}
	}
}
