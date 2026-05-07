package embedding

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"
)

const (
	voyageURL   = "https://api.voyageai.com/v1/embeddings"
	voyageModel = "voyage-3-lite"
	// Voyage AI aceita até 128 inputs por requisição
	maxBatchSize = 128
)

type Client struct {
	apiKey     string
	url        string
	httpClient *http.Client
}

func New(apiKey string) *Client {
	return &Client{
		apiKey: apiKey,
		url:    voyageURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Embed satisfaz a assinatura func([]string) ([][]float32, error) esperada pelo chunker.
// Divide automaticamente em batches quando necessário.
func (c *Client) Embed(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	var all [][]float32
	for i := 0; i < len(texts); i += maxBatchSize {
		end := min(i+maxBatchSize, len(texts))
		batch, err := c.embedBatch(texts[i:end])
		if err != nil {
			return nil, fmt.Errorf("batch %d-%d: %w", i, end, err)
		}
		all = append(all, batch...)
	}

	return all, nil
}

type voyageRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type voyageResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (c *Client) embedBatch(texts []string) ([][]float32, error) {
	const maxRetries = 7
	wait := 20 * time.Second

	for attempt := range maxRetries {
		result, err := c.doRequest(texts)
		if err != nil {
			return nil, err
		}
		if result != nil {
			return result, nil
		}
		// 429: backoff exponencial (20s, 40s, 80s, 160s, 320s, 640s)
		if attempt < maxRetries-1 {
			fmt.Fprintf(os.Stderr, "rate limit, aguardando %v...\n", wait)
			time.Sleep(wait)
			wait *= 2
		}
	}

	return nil, fmt.Errorf("voyage api: rate limit após %d tentativas", maxRetries)
}

func (c *Client) doRequest(texts []string) ([][]float32, error) {
	body, err := json.Marshal(voyageRequest{Input: texts, Model: voyageModel})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// nil = deve tentar novamente
	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, nil
	}

	var result voyageResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if result.Error != nil {
		return nil, fmt.Errorf("voyage api: %s", result.Error.Message)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("voyage api: status %d", resp.StatusCode)
	}

	embeddings := make([][]float32, len(texts))
	for _, d := range result.Data {
		embeddings[d.Index] = d.Embedding
	}

	return embeddings, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
