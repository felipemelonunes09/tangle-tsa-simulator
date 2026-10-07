# Tangle TSA Simulator

Simulador científico reprodutível de uma **abstração histórica da Tangle/IOTA 1.0** para comparar URTS, MCMC/WRW e SB-WRW. Os perfis PACTCare, TLIP e ElaadNL são **cenários sintéticos inspirados nas características operacionais dos casos de uso**; não são traces dessas plataformas.

## Documentação

A documentação pública está organizada em [`docs/README.md`](docs/README.md). Ela contém o guia de reprodução, a metodologia executável, as tabelas de resultados, a rastreabilidade do SB-WRW, as limitações e a auditoria entre o manuscrito e os artefatos.

Para começar:

- [reproduzir e verificar o experimento](docs/reproducibility.md);
- [entender a metodologia e as métricas](docs/methodology.md);
- [consultar os resultados canônicos](docs/results.md);
- [verificar divergências que ainda precisam ser corrigidas no manuscrito](docs/manuscript_consistency_audit.md).

## Execução que reproduz os resultados do artigo

A tabela numérica do manuscrito é reproduzida por
[`configs/article.yaml`](configs/article.yaml)

## Instalação

Requer Python 3.12 ou posterior. As versões estão fixadas em `requirements-lock.txt`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Para desenvolvimento sem instalação editável, prefixe os comandos com `PYTHONPATH=src`.

## Comandos

```bash
python -m tangle_sim validate --config configs/article.yaml
python -m tangle_sim run --config configs/article.yaml --output results/article
python scripts/assess_precision.py --input results/article/summary/metrics_summary.csv --attack-runs results/article/raw/attack_runs.csv --output results/article/summary/precision_assessment.csv
python scripts/verify_article_alignment.py --config configs/article.yaml --results results/article
pytest -q
```

Atalhos equivalentes: `make validate`, `make test`, `make article`,
`make article-check`, `make article-report` e `make article-audit`. `make paper`
permanece como alias de compatibilidade para `make article`. `--replicates` e
`--workers` sobrescrevem os controles de execução. Não use esses overrides para
produzir os números do artigo. O paralelismo é por réplica; cada worker recebe
streams imutáveis próprios e a redução final é ordenada.

O comando `run` executa baterias benignas e de ataque separadas, grava resultados temporários por réplica, reduz em ordem determinística, produz os CSVs/Parquet, manifesto, comparações pareadas e figuras. Um diretório de saída não vazio é recusado para evitar sobrescrever resultados preregistrados.

## Estrutura

- `src/tangle_sim/domain`: transações, conflitos, DAG, snapshots e peso acumulado;
- `src/tangle_sim/algorithms`: URTS, MCMC/WRW e `OperationalSBWRW`;
- `src/tangle_sim/simulation`: RNG, workloads e motores benigno/ataque;
- `src/tangle_sim/metrics`: confiança, abandono, bootstrap, Kaplan–Meier e RMST;
- `src/tangle_sim/reporting`: tabelas, comparações e gráficos;
- `configs/article.yaml`: configuração final autocontida, incluindo os três cenários;
- [`docs`](docs/README.md): portal da documentação, metodologia, reprodução, resultados, rastreabilidade e limitações;
- `tests`: testes unitários, estatísticos, de referência e integração.

## Reprodutibilidade e interpretação

O manifesto registra hash SHA-256 da configuração resolvida, hash da especificação normativa SB-WRW, seeds de raiz, dependências, sistema, definições das métricas e estado Git. Cada linha bruta registra os substreams. Workload, emissores, setores, atrasos e agenda adversarial são comuns aos três algoritmos na mesma réplica; seleção e medição usam streams separados.

## Referências centrais

- Ahmed et al., “SB-WRW: Balancing fairness and security in IoT-adapted tip selection for the tangle”, DOI [10.1016/j.iot.2025.101819](https://doi.org/10.1016/j.iot.2025.101819).
- Kuśmierz et al., [“Properties of the Tangle for Uniform Random and Random Walk Tip Selection”](https://arxiv.org/abs/2001.07734), IEEE Blockchain 2019.
- Ferraro, King e Shorten, [“On the stability of unverified transactions in a DAG-based Distributed Ledger”](https://arxiv.org/abs/1901.07302), IEEE TAC 2020.
- Penzkofer et al., [“Parasite Chain Detection in the IOTA Protocol”](https://doi.org/10.4230/OASIcs.Tokenomics.2020.8).
- Popov, “The Tangle”, documento técnico histórico; não descreve a rede IOTA atual.
