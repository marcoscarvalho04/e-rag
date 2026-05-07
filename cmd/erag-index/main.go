package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/config"
	"github.com/mpsiqueira/e-rag/internal/embedding"
	"github.com/mpsiqueira/e-rag/internal/retriever"
)

func main() {
	config.Load(".env")

	input := flag.String("input", "", "arquivo de texto para indexar (obrigatório)")
	output := flag.String("output", "index.gob", "caminho do arquivo de índice gerado")
	k := flag.Int("k", 5, "número de vizinhos K-NN por chunk")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "uso: erag-index -input arquivo.txt [-output index.gob] [-k 5]")
		os.Exit(1)
	}

	apiKey := os.Getenv("VOYAGE_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "erro: VOYAGE_API_KEY não configurada")
		os.Exit(1)
	}

	text, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "erro lendo arquivo: %v\n", err)
		os.Exit(1)
	}

	client := embedding.New(apiKey)

	fmt.Printf("chunking semântico de %q...\n", *input)
	c := chunker.New()
	chunks, err := c.Chunk(string(text), client.Embed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "erro no chunking: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("%d chunks detectados\n", len(chunks))

	texts := make([]string, len(chunks))
	for i, ch := range chunks {
		texts[i] = ch.Text
	}

	fmt.Println("gerando embeddings...")
	embeddings, err := client.Embed(texts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "erro no embedding: %v\n", err)
		os.Exit(1)
	}

	kv := *k
	if kv >= len(chunks) {
		kv = len(chunks) - 1
	}

	fmt.Println("construindo índice...")
	idx, err := retriever.Build(chunks, embeddings, kv, client.Embed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "erro construindo índice: %v\n", err)
		os.Exit(1)
	}

	if err := idx.Save(*output); err != nil {
		fmt.Fprintf(os.Stderr, "erro salvando índice: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("índice salvo em %q\n", *output)
}
