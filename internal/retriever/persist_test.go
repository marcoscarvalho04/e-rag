package retriever

import (
	"path/filepath"
	"testing"
)

func TestSaveLoad_RoundTrip(t *testing.T) {
	idx := buildTestIndex(t)

	path := filepath.Join(t.TempDir(), "index.gob")

	if err := idx.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load(path, fakeEmbedder)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// chunks preservados
	if len(loaded.chunks) != len(idx.chunks) {
		t.Errorf("chunks: got %d, want %d", len(loaded.chunks), len(idx.chunks))
	}
	for i, c := range loaded.chunks {
		if c.Text != idx.chunks[i].Text {
			t.Errorf("chunk[%d].Text diverge após load", i)
		}
	}

	// grafo preservado
	if len(loaded.graph.Nodes) != len(idx.graph.Nodes) {
		t.Errorf("nodes: got %d, want %d", len(loaded.graph.Nodes), len(idx.graph.Nodes))
	}
	if len(loaded.graph.Edges) != len(idx.graph.Edges) {
		t.Errorf("edges: got %d, want %d", len(loaded.graph.Edges), len(idx.graph.Edges))
	}

	// comunidades preservadas
	if len(loaded.community.NodeCommunity) != len(idx.community.NodeCommunity) {
		t.Errorf("NodeCommunity: got %d, want %d",
			len(loaded.community.NodeCommunity), len(idx.community.NodeCommunity))
	}
	for i, c := range loaded.community.NodeCommunity {
		if c != idx.community.NodeCommunity[i] {
			t.Errorf("NodeCommunity[%d]: got %d, want %d", i, c, idx.community.NodeCommunity[i])
		}
	}
}

func TestSaveLoad_SearchConsistent(t *testing.T) {
	idx := buildTestIndex(t)
	path := filepath.Join(t.TempDir(), "index.gob")

	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}

	loaded, err := Load(path, fakeEmbedder)
	if err != nil {
		t.Fatal(err)
	}

	// resultado de busca deve ser idêntico antes e depois do load
	query := "qual o impacto dos juros no banco?"

	r1, err := idx.Search(query, 5)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := loaded.Search(query, 5)
	if err != nil {
		t.Fatal(err)
	}

	if r1.Anchor.NodeID != r2.Anchor.NodeID {
		t.Errorf("âncora diverge: original=%d loaded=%d", r1.Anchor.NodeID, r2.Anchor.NodeID)
	}
	if r1.Anchor.CommunityID != r2.Anchor.CommunityID {
		t.Errorf("communityID diverge: original=%d loaded=%d",
			r1.Anchor.CommunityID, r2.Anchor.CommunityID)
	}
}

func TestLoad_FileNotFound(t *testing.T) {
	_, err := Load("/nao/existe/index.gob", fakeEmbedder)
	if err == nil {
		t.Error("esperava erro ao carregar arquivo inexistente")
	}
}

func TestSave_InvalidPath(t *testing.T) {
	idx := buildTestIndex(t)
	err := idx.Save("/nao/existe/diretorio/index.gob")
	if err == nil {
		t.Error("esperava erro ao salvar em caminho inválido")
	}
}

func TestSaveLoad_EmbedsOnlyQuery(t *testing.T) {
	// garante que Load não chama embedder para os chunks —
	// apenas para queries em tempo de retrieval
	calls := 0
	countingEmbedder := func(texts []string) ([][]float32, error) {
		calls++
		return fakeEmbedder(texts)
	}

	idx := buildTestIndex(t)
	path := filepath.Join(t.TempDir(), "index.gob")
	idx.Save(path)

	calls = 0 // reset após build
	loaded, err := Load(path, countingEmbedder)
	if err != nil {
		t.Fatal(err)
	}

	if calls != 0 {
		t.Errorf("Load não deveria chamar embedder, mas chamou %d vez(es)", calls)
	}

	loaded.Search("juros banco crédito", 3)
	if calls != 1 {
		t.Errorf("Search deveria chamar embedder 1 vez, chamou %d", calls)
	}
}

