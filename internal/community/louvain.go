package community

import (
	"github.com/mpsiqueira/e-rag/internal/graph"
)

// Result mapeia cada nó à sua comunidade final.
type Result struct {
	// NodeCommunity[i] = ID da comunidade do nó i
	NodeCommunity []int
	// Members[c] = lista de nós que pertencem à comunidade c
	Members map[int][]uint32
}

// Detect executa o algoritmo de Louvain sobre o grafo e retorna as comunidades.
func Detect(g *graph.Graph) *Result {
	adj := symmetrize(g)
	n := len(g.Nodes)

	degree := make([]float64, n)
	var m float64
	for i := 0; i < n; i++ {
		for _, e := range adj[i] {
			degree[i] += float64(e.Weight)
		}
		m += degree[i]
	}
	m /= 2.0

	if m == 0 {
		// grafo sem arestas: cada nó é sua própria comunidade
		return buildResult(makeRange(n), n)
	}

	community := make([]int, n)
	for i := range community {
		community[i] = i
	}

	sigmaTot := make([]float64, n)
	for i := 0; i < n; i++ {
		sigmaTot[i] = degree[i]
	}
	sigmaIn := make([]float64, n)

	for {
		moved := false

		for i := 0; i < n; i++ {
			currentComm := community[i]

			// remove i da comunidade atual para avaliar realocação
			kiInCurrent := edgeWeightToComm(adj[i], community, currentComm, i)
			sigmaIn[currentComm] -= 2 * kiInCurrent
			sigmaTot[currentComm] -= degree[i]

			// baseline: ganho de retornar à comunidade atual
			bestComm := currentComm
			bestGain := kiInCurrent/m - sigmaTot[currentComm]*degree[i]/(2*m*m)

			// avalia comunidades vizinhas — só move se for estritamente melhor
			visited := map[int]bool{currentComm: true}
			for _, e := range adj[i] {
				c := community[e.To]
				if visited[c] {
					continue
				}
				visited[c] = true

				kiInC := edgeWeightToComm(adj[i], community, c, i)
				gain := kiInC/m - sigmaTot[c]*degree[i]/(2*m*m)
				if gain > bestGain {
					bestGain = gain
					bestComm = c
				}
			}

			// reinsere i na melhor comunidade encontrada
			community[i] = bestComm
			kiInBest := edgeWeightToComm(adj[i], community, bestComm, i)
			sigmaIn[bestComm] += 2 * kiInBest
			sigmaTot[bestComm] += degree[i]

			if bestComm != currentComm {
				moved = true
			}
		}

		if !moved {
			break
		}
	}

	return buildResult(community, n)
}

// edgeWeightToComm soma os pesos das arestas de node para nós na comunidade c,
// excluindo o próprio node.
func edgeWeightToComm(edges []graph.Edge, community []int, c int, self int) float64 {
	var total float64
	for _, e := range edges {
		if int(e.To) != self && community[e.To] == c {
			total += float64(e.Weight)
		}
	}
	return total
}

// symmetrize constrói lista de adjacência simétrica a partir do grafo K-NN direcional.
func symmetrize(g *graph.Graph) [][]graph.Edge {
	type key struct{ a, b uint32 }
	weights := make(map[key]float32)

	for i, edges := range g.Edges {
		for _, e := range edges {
			a, b := uint32(i), e.To
			if a > b {
				a, b = b, a
			}
			k := key{a, b}
			if e.Weight > weights[k] {
				weights[k] = e.Weight
			}
		}
	}

	n := len(g.Nodes)
	adj := make([][]graph.Edge, n)
	for k, w := range weights {
		adj[k.a] = append(adj[k.a], graph.Edge{To: k.b, Weight: w})
		adj[k.b] = append(adj[k.b], graph.Edge{To: k.a, Weight: w})
	}

	return adj
}

// buildResult consolida o mapeamento final com IDs contíguos (0, 1, 2...).
func buildResult(community []int, n int) *Result {
	remap := make(map[int]int)
	members := make(map[int][]uint32)
	nextID := 0

	nodeCommunity := make([]int, n)
	for i, c := range community {
		if _, ok := remap[c]; !ok {
			remap[c] = nextID
			nextID++
		}
		mapped := remap[c]
		nodeCommunity[i] = mapped
		members[mapped] = append(members[mapped], uint32(i))
	}

	return &Result{NodeCommunity: nodeCommunity, Members: members}
}

func makeRange(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = i
	}
	return s
}
