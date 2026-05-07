package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"

	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/config"
	"github.com/mpsiqueira/e-rag/internal/embedding"
	"github.com/mpsiqueira/e-rag/internal/indexmgr"
	"github.com/mpsiqueira/e-rag/internal/retriever"
)

func main() {
	// usado pelo HEALTHCHECK do Docker — sem curl no distroless
	if len(os.Args) == 2 && os.Args[1] == "-healthcheck" {
		resp, err := http.Get("http://localhost" + envOr("ERAG_ADDR", ":8080") + "/health")
		if err != nil || resp.StatusCode != http.StatusOK {
			os.Exit(1)
		}
		os.Exit(0)
	}

	config.Load(".env")

	apiKey := os.Getenv("VOYAGE_API_KEY")
	if apiKey == "" {
		slog.Error("VOYAGE_API_KEY não configurada")
		os.Exit(1)
	}

	indexDir := envOr("ERAG_INDEX_DIR", "./indexes")
	maxIndexes, _ := strconv.Atoi(envOr("ERAG_MAX_INDEXES", "10"))
	addr := envOr("ERAG_ADDR", ":8080")

	client := embedding.New(apiKey)

	mgr, err := indexmgr.New(indexDir, maxIndexes, client.Embed)
	if err != nil {
		slog.Error("inicializando gerenciador de índices", "err", err)
		os.Exit(1)
	}

	srv := &server{mgr: mgr, embedder: client.Embed}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", srv.health)
	mux.HandleFunc("GET /indexes", srv.listIndexes)
	mux.HandleFunc("POST /indexes/{name}", srv.createIndex)
	mux.HandleFunc("DELETE /indexes/{name}", srv.deleteIndex)
	mux.HandleFunc("POST /indexes/{name}/search", srv.search)

	slog.Info("e-rag servidor iniciado", "addr", addr, "index_dir", indexDir)
	if err := http.ListenAndServe(addr, mux); err != nil {
		slog.Error("servidor encerrado", "err", err)
		os.Exit(1)
	}
}

type server struct {
	mgr      *indexmgr.Manager
	embedder func([]string) ([][]float32, error)
}

// --- handlers ---

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *server) listIndexes(w http.ResponseWriter, r *http.Request) {
	names, err := s.mgr.List()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if names == nil {
		names = []string{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"indexes": names})
}

type createRequest struct {
	Documents []string `json:"documents"`
	K         int      `json:"k"`
}

func (s *server) createIndex(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")

	var req createRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("body inválido: %w", err))
		return
	}
	if len(req.Documents) == 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("documents não pode ser vazio"))
		return
	}
	if req.K <= 0 {
		req.K = 5
	}

	// chunking semântico de todos os documentos
	c := chunker.New()
	var allChunks []chunker.Chunk
	for _, doc := range req.Documents {
		chunks, err := c.Chunk(doc, s.embedder)
		if err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Errorf("chunking: %w", err))
			return
		}
		allChunks = append(allChunks, chunks...)
	}

	if len(allChunks) == 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("nenhum chunk gerado dos documentos"))
		return
	}

	// embed todos os chunks em batch único
	texts := make([]string, len(allChunks))
	for i, ch := range allChunks {
		texts[i] = ch.Text
	}
	embeddings, err := s.embedder(texts)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("embedding: %w", err))
		return
	}

	k := req.K
	if k >= len(allChunks) {
		k = len(allChunks) - 1
	}

	idx, err := retriever.Build(allChunks, embeddings, k, s.embedder)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("construindo índice: %w", err))
		return
	}

	if err := s.mgr.Put(name, idx); err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("salvando índice: %w", err))
		return
	}

	slog.Info("índice criado", "name", name, "chunks", len(allChunks))
	writeJSON(w, http.StatusCreated, map[string]any{
		"name":   name,
		"chunks": len(allChunks),
	})
}

func (s *server) deleteIndex(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if err := s.mgr.Delete(name); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

type searchRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

type searchResponse struct {
	Anchor    chunkResult   `json:"anchor"`
	Community []chunkResult `json:"community"`
}

type chunkResult struct {
	Text        string  `json:"text"`
	Score       float32 `json:"score"`
	CommunityID int     `json:"community_id"`
}

func (s *server) search(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")

	var req searchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("body inválido: %w", err))
		return
	}
	if req.Query == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("query não pode ser vazia"))
		return
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}

	idx, err := s.mgr.Get(name)
	if err != nil {
		var notFound indexmgr.ErrNotFound
		if errors.As(err, &notFound) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	result, err := idx.Search(req.Query, req.TopK)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("busca: %w", err))
		return
	}
	if result == nil {
		writeJSON(w, http.StatusOK, searchResponse{Community: []chunkResult{}})
		return
	}

	resp := searchResponse{
		Anchor: chunkResult{
			Text:        result.Anchor.Text,
			Score:       result.Anchor.Score,
			CommunityID: result.Anchor.CommunityID,
		},
		Community: make([]chunkResult, len(result.Community)),
	}
	for i, c := range result.Community {
		resp.Community[i] = chunkResult{
			Text:        c.Text,
			Score:       c.Score,
			CommunityID: c.CommunityID,
		}
	}

	writeJSON(w, http.StatusOK, resp)
}

// --- helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, err error) {
	slog.Error("request error", "status", status, "err", err)
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
