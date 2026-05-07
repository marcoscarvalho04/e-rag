// Package erag implementa Retrieval Augmented Generation com preservação de
// contexto semântico via grafo K-NN e detecção de comunidades (Louvain).
//
// Ao contrário do RAG clássico, que trata chunks como vetores independentes,
// o erag constrói um grafo de afinidade semântica entre chunks e usa detecção
// de comunidades para retornar regiões coerentes do conhecimento — não
// fragmentos isolados.
//
// Uso básico:
//
//	embedder := func(texts []string) ([][]float32, error) {
//	    // qualquer modelo de embedding
//	}
//
//	idx, err := erag.New([]string{"documento 1", "documento 2"}, embedder)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	result, err := idx.Search("minha query", 5)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	fmt.Println(result.Anchor.Text)
//	for _, c := range result.Community {
//	    fmt.Println(c.Text, c.Score)
//	}
package erag

import (
	"fmt"

	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/retriever"
)

// retrieverResult é um alias interno para evitar repetição de tipo.
type retrieverResult = retriever.SearchResult

// Embedder é a função responsável por transformar textos em vetores.
// Qualquer modelo de embedding pode ser usado — local ou via API.
type Embedder func(texts []string) ([][]float32, error)

// Chunk é um fragmento semântico do conhecimento indexado.
type Chunk struct {
	// Text é o conteúdo textual do chunk.
	Text string
	// Score é a similaridade coseno com a query (0 a 1).
	Score float32
	// CommunityID identifica a comunidade semântica à qual o chunk pertence.
	CommunityID int
}

// Result é o resultado de uma busca.
type Result struct {
	// Anchor é o chunk mais próximo semanticamente da query.
	Anchor Chunk
	// Community são os vizinhos semânticos do âncora na mesma comunidade,
	// ordenados por relevância decrescente.
	Community []Chunk
}

// Options configura o comportamento do pipeline de indexação.
type Options struct {
	// K define quantos vizinhos cada chunk conecta no grafo K-NN.
	// Valor maior captura mais relações mas aumenta o custo de indexação.
	// Padrão: 5.
	K int
	// WindowSize define o tamanho de cada lado da janela deslizante
	// usada para detectar fronteiras semânticas entre sentenças.
	// Padrão: 3.
	WindowSize int
	// BreakThreshold controla a sensibilidade da detecção de fronteiras.
	// Valores menores geram chunks maiores; maiores geram chunks menores.
	// Padrão: 0.25.
	BreakThreshold float64
}

// Option é uma função que configura Options.
type Option func(*Options)

// WithK configura o número de vizinhos K-NN por chunk.
func WithK(k int) Option {
	return func(o *Options) { o.K = k }
}

// WithWindowSize configura o tamanho da janela deslizante do chunker.
func WithWindowSize(n int) Option {
	return func(o *Options) { o.WindowSize = n }
}

// WithBreakThreshold configura a sensibilidade do chunker semântico.
func WithBreakThreshold(t float64) Option {
	return func(o *Options) { o.BreakThreshold = t }
}

func defaultOptions() *Options {
	return &Options{
		K:              5,
		WindowSize:     3,
		BreakThreshold: 0.25,
	}
}

// Index é o índice semântico construído sobre uma coleção de documentos.
// É seguro para uso concorrente após construção.
type Index struct {
	internal *retriever.Index
}

// New constrói um índice a partir de uma lista de documentos em texto.
// Cada documento é dividido em chunks semânticos, embedado e indexado
// no grafo K-NN com detecção de comunidades via Louvain.
//
// O embedder recebe listas de textos e deve retornar um embedding por texto,
// na mesma ordem. Qualquer modelo de embedding pode ser usado.
func New(documents []string, embedder Embedder, opts ...Option) (*Index, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(o)
	}

	c := &chunker.Chunker{
		WindowSize:     o.WindowSize,
		BreakThreshold: o.BreakThreshold,
	}

	if len(documents) == 0 {
		return nil, fmt.Errorf("erag: documents não pode ser vazio")
	}

	var allChunks []chunker.Chunk
	for _, doc := range documents {
		chunks, err := c.Chunk(doc, embedder)
		if err != nil {
			return nil, err
		}
		allChunks = append(allChunks, chunks...)
	}

	if len(allChunks) == 0 {
		return nil, fmt.Errorf("erag: nenhum chunk gerado dos documentos")
	}

	texts := make([]string, len(allChunks))
	for i, ch := range allChunks {
		texts[i] = ch.Text
	}

	embeddings, err := embedder(texts)
	if err != nil {
		return nil, err
	}

	k := o.K
	if k >= len(allChunks) {
		k = len(allChunks) - 1
	}

	idx, err := retriever.Build(allChunks, embeddings, k, embedder)
	if err != nil {
		return nil, err
	}

	return &Index{internal: idx}, nil
}

// Load restaura um índice previamente salvo com Save.
// O embedder deve ser o mesmo tipo usado na construção — os embeddings
// dos chunks não são recalculados, apenas o da query em cada Search.
func Load(path string, embedder Embedder) (*Index, error) {
	idx, err := retriever.Load(path, embedder)
	if err != nil {
		return nil, err
	}
	return &Index{internal: idx}, nil
}

// Save persiste o índice em disco no formato gob.
// O embedder não é serializado — deve ser fornecido novamente em Load.
func (idx *Index) Save(path string) error {
	return idx.internal.Save(path)
}

// SearchByVector executa o retrieval com um embedding já computado.
// Útil quando o embedding da query já foi calculado externamente,
// evitando uma chamada extra ao embedder.
func (idx *Index) SearchByVector(queryVec []float32, topK int) (*Result, error) {
	r, err := idx.internal.SearchByVector(queryVec, topK)
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, nil
	}
	return toResult(r), nil
}

// Search encontra o chunk mais relevante para a query e retorna
// o chunk âncora junto com os vizinhos semânticos da sua comunidade.
//
// topK limita o número de vizinhos retornados em Result.Community.
// Use 0 para retornar todos os membros da comunidade.
func (idx *Index) Search(query string, topK int) (*Result, error) {
	r, err := idx.internal.Search(query, topK)
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, nil
	}
	return toResult(r), nil
}

func toResult(r *retrieverResult) *Result {
	result := &Result{
		Anchor: Chunk{
			Text:        r.Anchor.Text,
			Score:       r.Anchor.Score,
			CommunityID: r.Anchor.CommunityID,
		},
		Community: make([]Chunk, len(r.Community)),
	}
	for i, c := range r.Community {
		result.Community[i] = Chunk{
			Text:        c.Text,
			Score:       c.Score,
			CommunityID: c.CommunityID,
		}
	}
	return result
}
