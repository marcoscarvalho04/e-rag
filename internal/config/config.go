package config

import (
	"bufio"
	"os"
	"strings"
)

// Load carrega variáveis do arquivo .env se ainda não estiverem no ambiente.
// Seguro chamar mesmo se .env não existir.
func Load(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		// só define se não estiver já no ambiente
		if os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}
}
