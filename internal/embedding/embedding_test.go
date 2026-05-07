package embedding

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbed_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req voyageRequest
		json.NewDecoder(r.Body).Decode(&req)

		data := make([]struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}, len(req.Input))

		for i := range req.Input {
			data[i].Index = i
			data[i].Embedding = []float32{float32(i), float32(i + 1)}
		}

		json.NewEncoder(w).Encode(voyageResponse{Data: data})
	}))
	defer server.Close()

	c := newWithURL("fake-key", server.URL+"/v1/embeddings")

	texts := []string{"texto um", "texto dois", "texto três"}
	got, err := c.Embed(texts)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(texts) {
		t.Errorf("esperava %d embeddings, got %d", len(texts), len(got))
	}

	// verifica ordem preservada
	for i, emb := range got {
		if emb[0] != float32(i) {
			t.Errorf("embedding[%d] fora de ordem: esperava %f, got %f", i, float32(i), emb[0])
		}
	}
}

func TestEmbed_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(voyageResponse{
			Error: &struct {
				Message string `json:"message"`
			}{Message: "invalid api key"},
		})
	}))
	defer server.Close()

	c := newWithURL("bad-key", server.URL+"/v1/embeddings")
	_, err := c.Embed([]string{"texto"})
	if err == nil {
		t.Error("esperava erro de autenticação")
	}
}

func TestEmbed_Batching(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req voyageRequest
		json.NewDecoder(r.Body).Decode(&req)

		data := make([]struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}, len(req.Input))
		for i := range req.Input {
			data[i].Index = i
			data[i].Embedding = []float32{1.0}
		}
		json.NewEncoder(w).Encode(voyageResponse{Data: data})
	}))
	defer server.Close()

	c := newWithURL("fake-key", server.URL+"/v1/embeddings")

	// força 2 batches: 128 + 10
	texts := make([]string, 138)
	for i := range texts {
		texts[i] = "texto"
	}

	got, err := c.Embed(texts)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != 138 {
		t.Errorf("esperava 138 embeddings, got %d", len(got))
	}

	if calls != 2 {
		t.Errorf("esperava 2 chamadas à API (batching), got %d", calls)
	}
}

func TestEmbed_Empty(t *testing.T) {
	c := New("fake-key")
	got, err := c.Embed(nil)
	if err != nil || got != nil {
		t.Errorf("embed vazio deveria retornar nil, nil")
	}
}

// newWithURL permite injetar URL do servidor de teste
func newWithURL(apiKey, url string) *Client {
	c := New(apiKey)
	// sobrescreve a constante via campo interno
	c.url = url
	return c
}
