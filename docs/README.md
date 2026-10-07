# Documentação do Tangle TSA Simulator

Este diretório é a referência técnica do repositório. O experimento público possui uma única configuração executável, [`configs/article.yaml`](../configs/article.yaml). Todos os resultados são gerados localmente e o diretório `results/` não é versionado.

O simulador modela uma abstração histórica da Tangle/IOTA 1.0. PACTCare, TLIP e ElaadNL são perfis sintéticos inspirados em características operacionais; nenhum deles deve ser descrito como reprodução ou trace da plataforma real.

## Navegação

| Documento | Conteúdo |
|---|---|
| [Reprodução](reproducibility.md) | Instalação, execução, verificação e estrutura dos artefatos |
| [Metodologia](methodology.md) | Modelo, cenários, algoritmos, baterias, métricas e estatística |
| [Resultados](results.md) | Valores canônicos apresentados no manuscrito e interpretação |
| [Auditoria manuscrito–artefatos](manuscript_consistency_audit.md) | Divergências objetivas que precisam ser corrigidas antes da publicação |
| [Rastreabilidade do SB-WRW](sb_wrw_traceability.md) | Correspondência entre a especificação operacional, o código e os testes |
| [Revisão da implementação](algorithm_implementation_review.md) | Evidências automatizadas sobre URTS, MCMC e SB-WRW |
| [Limitações](limitations.md) | Limites de validade e afirmações que os dados não sustentam |

## Estado canônico

- Configuração: [`configs/article.yaml`](../configs/article.yaml)
- Hash SHA-256 da configuração resolvida: `7c675825a7ee13a1efd6e3659afe1e81ba2b5cfc2cdda04e51bc6827c537a87c`
- Destino local padrão: `results/article/` (ignorado pelo Git)
- Versão do simulador: `0.7.0`
- Seed raiz: `20250701`
- Réplicas: 10 por combinação de cenário, algoritmo e bateria
- Execuções brutas: 90 benignas e 90 adversariais

As médias das 27 células principais são verificadas automaticamente com a precisão exibida no artigo. O auditor também valida contagens, censura, streams aleatórios comuns, artefatos obrigatórios e hashes do conjunto congelado.

## Verificação rápida

Execute a partir da raiz do repositório:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
make validate
make test
make article
```

`make article` executa as réplicas, calcula a precisão e verifica os resultados gerados em `results/article`. Depois disso, `make article-check` pode repetir somente a verificação local.

## Citação

Os metadados de citação do software estão em [`CITATION.cff`](../CITATION.cff). Ao reutilizar o simulador, cite o software e o artigo associado.

[Voltar ao README principal](../README.md)
