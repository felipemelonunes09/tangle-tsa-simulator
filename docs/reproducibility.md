# Guia de reprodução

Este guia descreve a execução pública que gera os números do manuscrito. Todos os comandos partem da raiz do repositório.

## Requisitos e instalação

- Python 3.12 ou posterior;
- dependências nas versões exatas de `requirements-lock.txt`;
- espaço disponível para CSV, Parquet e figuras gerados durante a execução.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Sem instalação editável, acrescente `PYTHONPATH=src` aos comandos Python.

## Verificar código e configuração

```bash
python -m tangle_sim validate --config configs/article.yaml
pytest -q
```

Os atalhos equivalentes são `make validate` e `make test`. O repositório não distribui resultados prontos: todo o diretório `results/` é ignorado pelo Git.

A verificação canônica exige:

- hash da configuração resolvida esperado;
- 10 réplicas e seed raiz `20250701`;
- parâmetros temporais, `alpha` e agenda adversarial esperados;
- 27 médias principais iguais aos valores do artigo na precisão publicada;
- contagens esperadas de eventos adversariais;
- 90 linhas benignas, 90 adversariais e 45 linhas no resumo;
- consistência entre CSV e Parquet;
- figura principal e avaliação de precisão presentes;
- consistência semântica dos resultados gerados com os valores documentados.

Uma execução bem-sucedida imprime JSON com `"aligned": true`.

## Reexecutar o experimento

```bash
python -m tangle_sim run \
  --config configs/article.yaml \
  --output results/article

python scripts/assess_precision.py \
  --input results/article/summary/metrics_summary.csv \
  --attack-runs results/article/raw/attack_runs.csv \
  --output results/article/summary/precision_assessment.csv

python scripts/verify_article_alignment.py \
  --config configs/article.yaml \
  --results results/article
```

`make article` executa os três comandos. O diretório de saída deve estar vazio ou conter checkpoints compatíveis; essa restrição evita sobrescrever silenciosamente uma execução finalizada. `--replicates` e `--workers` existem para diagnóstico, mas não devem ser usados na reprodução dos números publicados.

## Estrutura de uma execução

```text
results/<execucao>/
├── config_resolved.yaml
├── manifest.json
├── environment.txt
├── audit.json
├── raw/
│   ├── benign_runs.csv
│   ├── benign_runs.parquet
│   ├── attack_runs.csv
│   └── attack_runs.parquet
├── summary/
│   ├── metrics_summary.csv
│   ├── pairwise_comparisons.csv
│   ├── precision_assessment.csv
│   └── sensitivity_summary.csv
└── plots/
```

O manifesto registra versão do simulador, hash da configuração, dependências, plataforma, seeds, definições das métricas e conformidade declarada do SB-WRW. Os arquivos brutos preservam os identificadores dos substreams aleatórios usados por réplica.

## Determinismo e streams comuns

Na mesma combinação cenário–réplica, os algoritmos recebem o mesmo workload, topologia, atrasos e agenda adversarial. Seleção e medição usam streams próprios. A redução dos checkpoints é ordenada, de modo que a ordem de conclusão dos workers não altera a sequência das linhas consolidadas.

O verificador prioriza equivalência semântica dos valores, parâmetros e contagens, pois metadados voláteis e diferenças de bibliotecas podem impedir identidade byte a byte entre plataformas. Os hashes presentes nos artefatos locais servem à auditoria da própria execução, mas nenhum resultado é enviado ao Git.

## Regenerar somente figuras

Depois de uma execução em `results/article`:

```bash
python -m tangle_sim report --input results/article --language pt-BR
python -m tangle_sim audit --input results/article
```

Atalhos: `make article-report` e `make article-audit`.

[Voltar ao índice da documentação](README.md)
