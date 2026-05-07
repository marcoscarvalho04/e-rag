package indexmgr

import (
	"container/list"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/mpsiqueira/e-rag/internal/retriever"
)

// Manager mantém múltiplos índices em memória com eviction LRU.
type Manager struct {
	mu       sync.RWMutex
	entries  map[string]*entry
	lru      *list.List // frente = mais recente, fundo = candidato a eviction
	maxItems int
	dir      string
	embedder func([]string) ([][]float32, error)
}

type entry struct {
	name  string
	index *retriever.Index
	elem  *list.Element
}

// New cria um gerenciador que persiste índices em dir.
// maxItems define quantos índices cabem simultaneamente na memória.
func New(dir string, maxItems int, embedder func([]string) ([][]float32, error)) (*Manager, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("criando diretório de índices: %w", err)
	}
	return &Manager{
		entries:  make(map[string]*entry),
		lru:      list.New(),
		maxItems: maxItems,
		dir:      dir,
		embedder: embedder,
	}, nil
}

// Get retorna um índice pelo nome, carregando do disco se necessário.
func (m *Manager) Get(name string) (*retriever.Index, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if e, ok := m.entries[name]; ok {
		m.lru.MoveToFront(e.elem)
		return e.index, nil
	}

	// não está em memória — carrega do disco
	path := m.indexPath(name)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, ErrNotFound{Name: name}
	}

	idx, err := retriever.Load(path, m.embedder)
	if err != nil {
		return nil, fmt.Errorf("carregando índice %q: %w", name, err)
	}

	m.admit(name, idx)
	return idx, nil
}

// Put salva um índice em disco e o admite na memória.
func (m *Manager) Put(name string, idx *retriever.Index) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	path := m.indexPath(name)
	if err := idx.Save(path); err != nil {
		return fmt.Errorf("salvando índice %q: %w", name, err)
	}

	// atualiza entrada em memória se já existia
	if e, ok := m.entries[name]; ok {
		e.index = idx
		m.lru.MoveToFront(e.elem)
		return nil
	}

	m.admit(name, idx)
	return nil
}

// Delete remove um índice da memória e do disco.
func (m *Manager) Delete(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if e, ok := m.entries[name]; ok {
		m.lru.Remove(e.elem)
		delete(m.entries, name)
	}

	path := m.indexPath(name)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("removendo índice %q: %w", name, err)
	}

	return nil
}

// List retorna os nomes de todos os índices disponíveis em disco.
func (m *Manager) List() ([]string, error) {
	entries, err := os.ReadDir(m.dir)
	if err != nil {
		return nil, fmt.Errorf("listando índices: %w", err)
	}

	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".gob") {
			names = append(names, strings.TrimSuffix(e.Name(), ".gob"))
		}
	}
	return names, nil
}

// admit adiciona um índice na memória, evictando o LRU se necessário.
// deve ser chamado com m.mu já adquirido.
func (m *Manager) admit(name string, idx *retriever.Index) {
	if len(m.entries) >= m.maxItems {
		m.evict()
	}

	elem := m.lru.PushFront(name)
	m.entries[name] = &entry{name: name, index: idx, elem: elem}
}

// evict remove o índice menos recentemente usado da memória.
func (m *Manager) evict() {
	oldest := m.lru.Back()
	if oldest == nil {
		return
	}
	name := oldest.Value.(string)
	m.lru.Remove(oldest)
	delete(m.entries, name)
}

func (m *Manager) indexPath(name string) string {
	return filepath.Join(m.dir, name+".gob")
}

// ErrNotFound é retornado quando um índice não existe.
type ErrNotFound struct {
	Name string
}

func (e ErrNotFound) Error() string {
	return fmt.Sprintf("índice %q não encontrado", e.Name)
}
