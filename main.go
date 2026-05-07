package main

import (
	"fmt"
	"os"

	"github.com/mpsiqueira/e-rag/internal/chunker"
	"github.com/mpsiqueira/e-rag/internal/config"
	"github.com/mpsiqueira/e-rag/internal/embedding"
	"github.com/mpsiqueira/e-rag/internal/retriever"
)

const sampleText = `
O banco central elevou a taxa de juros básica da economia em 0,5 ponto percentual.
A decisão foi unânime entre os membros do comitê de política monetária.
O objetivo é conter a inflação que vem pressionando os preços ao consumidor.
O crédito deve ficar mais caro nos próximos meses como consequência direta da medida.
Analistas esperam que o ciclo de aperto monetário continue nas próximas reuniões.

A chuva voltou a cair com intensidade após semanas de estiagem na região sudeste.
Os reservatórios que abastecem as grandes cidades voltaram a subir de nível.
A temperatura caiu significativamente com a chegada das frentes frias do sul do país.
O vento forte derrubou árvores e causou falta de energia em alguns bairros.
Defesa civil emitiu alerta para risco de alagamentos nas zonas de baixada.
`

var queries = []string{
	"qual o impacto da taxa de juros na economia?",
	"como está o clima e a situação das chuvas?",
}

func main() {
	config.Load(".env")

	apiKey := os.Getenv("VOYAGE_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "erro: VOYAGE_API_KEY não configurada")
		os.Exit(1)
	}

	client := embedding.New(apiKey)

	fmt.Println("=== e-rag: pipeline completo ===")
	fmt.Println()

	// 1. chunking semântico — 1 chamada à API (todas as sentenças)
	fmt.Println("[1/3] chunking semântico...")
	c := chunker.New()
	chunks, err := c.Chunk(sampleText, client.Embed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "erro no chunking: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("      %d chunks detectados:\n", len(chunks))
	for i, ch := range chunks {
		preview := ch.Text
		if len(preview) > 80 {
			preview = preview[:80] + "..."
		}
		fmt.Printf("      [%d] %s\n", i, preview)
	}
	fmt.Println()

	// 2. embed chunks + queries numa única chamada — minimiza requests
	fmt.Println("[2/3] gerando embeddings (chunks + queries em batch único)...")
	texts := make([]string, len(chunks)+len(queries))
	for i, ch := range chunks {
		texts[i] = ch.Text
	}
	for i, q := range queries {
		texts[len(chunks)+i] = q
	}

	allEmbeddings, err := client.Embed(texts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "erro no embedding: %v\n", err)
		os.Exit(1)
	}

	chunkEmbeddings := allEmbeddings[:len(chunks)]
	queryEmbeddings := allEmbeddings[len(chunks):]
	fmt.Printf("      %d embeddings gerados (%d dimensões cada)\n", len(allEmbeddings), len(allEmbeddings[0]))
	fmt.Println()

	// 3. constrói índice — sem chamadas adicionais à API
	fmt.Println("[3/3] construindo índice e executando retrieval...")
	k := 3
	if len(chunks) <= k {
		k = len(chunks) - 1
	}

	// embedder que usa os embeddings já computados — zero chamadas extras à API
	cachedEmbedder := makeCachedEmbedder(texts, allEmbeddings)

	idx, err := retriever.Build(chunks, chunkEmbeddings, k, cachedEmbedder)
	if err != nil {
		fmt.Fprintf(os.Stderr, "erro ao construir índice: %v\n", err)
		os.Exit(1)
	}
	fmt.Println()

	// retrieval usando embeddings já computados
	fmt.Println("=== resultados ===")
	fmt.Println()

	for i, query := range queries {
		fmt.Printf("query: \"%s\"\n", query)

		result, err := idx.SearchByVector(queryEmbeddings[i], 3)
		if err != nil {
			fmt.Fprintf(os.Stderr, "erro na busca: %v\n", err)
			continue
		}
		if result == nil {
			fmt.Println("sem resultado.")
			continue
		}

		fmt.Printf("âncora (score=%.4f):\n  %s\n", result.Anchor.Score, result.Anchor.Text)

		if len(result.Community) > 0 {
			fmt.Println("comunidade semântica:")
			for _, r := range result.Community {
				fmt.Printf("  (score=%.4f) %s\n", r.Score, r.Text)
			}
		}
		fmt.Println()
	}
}

// makeCachedEmbedder retorna um embedder que resolve via cache local.
// Evita chamadas à API para textos já embedados.
func makeCachedEmbedder(texts []string, embeddings [][]float32) func([]string) ([][]float32, error) {
	cache := make(map[string][]float32, len(texts))
	for i, t := range texts {
		cache[t] = embeddings[i]
	}
	return func(inputs []string) ([][]float32, error) {
		result := make([][]float32, len(inputs))
		for i, t := range inputs {
			if emb, ok := cache[t]; ok {
				result[i] = emb
			}
		}
		return result, nil
	}
}
