package indexmgr

import (
	"testing"

	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/retriever"
)

func fakeEmbedder(texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range texts {
		result[i] = []float32{float32(i), float32(i + 1)}
	}
	return result, nil
}

func buildIndex(t *testing.T) *retriever.Index {
	t.Helper()
	chunks := []chunker.Chunk{
		{Text: "texto um", Start: 0, End: 0},
		{Text: "texto dois", Start: 1, End: 1},
		{Text: "texto três", Start: 2, End: 2},
	}
	embeddings, _ := fakeEmbedder([]string{"texto um", "texto dois", "texto três"})
	idx, err := retriever.Build(chunks, embeddings, 2, fakeEmbedder)
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

func TestPutGet(t *testing.T) {
	mgr, err := New(t.TempDir(), 10, fakeEmbedder)
	if err != nil {
		t.Fatal(err)
	}

	idx := buildIndex(t)
	if err := mgr.Put("test", idx); err != nil {
		t.Fatal(err)
	}

	got, err := mgr.Get("test")
	if err != nil {
		t.Fatal(err)
	}
	if got == nil {
		t.Error("esperava índice, got nil")
	}
}

func TestGet_NotFound(t *testing.T) {
	mgr, _ := New(t.TempDir(), 10, fakeEmbedder)

	_, err := mgr.Get("inexistente")
	if err == nil {
		t.Error("esperava ErrNotFound")
	}
	if _, ok := err.(ErrNotFound); !ok {
		t.Errorf("esperava ErrNotFound, got %T", err)
	}
}

func TestDelete(t *testing.T) {
	mgr, _ := New(t.TempDir(), 10, fakeEmbedder)
	idx := buildIndex(t)
	mgr.Put("test", idx)

	if err := mgr.Delete("test"); err != nil {
		t.Fatal(err)
	}

	_, err := mgr.Get("test")
	if _, ok := err.(ErrNotFound); !ok {
		t.Error("após Delete, esperava ErrNotFound")
	}
}

func TestDelete_Idempotent(t *testing.T) {
	mgr, _ := New(t.TempDir(), 10, fakeEmbedder)
	// deletar índice que não existe não deve retornar erro
	if err := mgr.Delete("inexistente"); err != nil {
		t.Errorf("Delete de inexistente deveria ser no-op, got: %v", err)
	}
}

func TestList(t *testing.T) {
	mgr, _ := New(t.TempDir(), 10, fakeEmbedder)
	idx := buildIndex(t)

	mgr.Put("alpha", idx)
	mgr.Put("beta", idx)

	names, err := mgr.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 {
		t.Errorf("esperava 2 índices, got %d", len(names))
	}
}

func TestLRU_Eviction(t *testing.T) {
	// maxItems=2: ao adicionar o terceiro, o menos usado deve ser evictado da memória
	mgr, _ := New(t.TempDir(), 2, fakeEmbedder)
	idx := buildIndex(t)

	mgr.Put("a", idx)
	mgr.Put("b", idx)
	mgr.Put("c", idx) // evicta "a" da memória

	// "a" ainda deve existir em disco — Get deve recarregá-lo
	got, err := mgr.Get("a")
	if err != nil {
		t.Errorf("'a' deveria ser recarregado do disco: %v", err)
	}
	if got == nil {
		t.Error("esperava índice 'a' após recarga")
	}
}

func TestGet_LoadsFromDisk(t *testing.T) {
	dir := t.TempDir()
	idx := buildIndex(t)

	// salva com um gerenciador
	mgr1, _ := New(dir, 10, fakeEmbedder)
	mgr1.Put("persistido", idx)

	// carrega com outro gerenciador (memória zerada)
	mgr2, _ := New(dir, 10, fakeEmbedder)
	got, err := mgr2.Get("persistido")
	if err != nil {
		t.Fatalf("esperava carregar do disco: %v", err)
	}
	if got == nil {
		t.Error("índice carregado do disco é nil")
	}
}
