# Resultados canônicos

Os valores abaixo são os valores esperados registrados pelo verificador do artigo. Eles podem ser regenerados com `make article`; os arquivos resultantes ficam somente em `results/article/` e não são versionados.

## Eficiência em condição benigna

| Cenário | Algoritmo | Aprovação (%) | Pontas abandonadas (%) |
|---|---|---:|---:|
| PACTCare | URTS | 95,04 | 4,96 |
| PACTCare | MCMC | 87,09 | 12,91 |
| PACTCare | SB-WRW | 94,34 | 5,66 |
| TLIP | URTS | 94,30 | 5,70 |
| TLIP | MCMC | 91,96 | 8,04 |
| TLIP | SB-WRW | 93,56 | 6,44 |
| ElaadNL | URTS | 95,96 | 4,04 |
| ElaadNL | MCMC | 91,40 | 8,60 |
| ElaadNL | SB-WRW | 95,04 | 4,96 |

URTS obteve a maior aprovação média nos três cenários. SB-WRW permaneceu entre URTS e MCMC, com distância de 0,70, 0,74 e 0,92 ponto percentual para URTS em PACTCare, TLIP e ElaadNL.

## Resistência na bateria adversarial

| Cenário | Algoritmo | RMST até fraude (s) | Eventos observados / 10 |
|---|---|---:|---:|
| PACTCare | URTS | 24,00 | 10 |
| PACTCare | MCMC | 65,45 | 10 |
| PACTCare | SB-WRW | 76,15 | 4 |
| TLIP | URTS | 25,95 | 10 |
| TLIP | MCMC | 81,75 | 10 |
| TLIP | SB-WRW | 98,75 | 2 |
| ElaadNL | URTS | 39,40 | 10 |
| ElaadNL | MCMC | 77,70 | 10 |
| ElaadNL | SB-WRW | 51,65 | 8 |

SB-WRW apresentou a maior RMST no PACTCare e no TLIP. No ElaadNL, MCMC apresentou a maior RMST. URTS foi o menor em todos os cenários. Como há censura, RMST e contagem de eventos devem ser interpretadas em conjunto.

Depois de `make article`, os valores completos, intervalos e campos estatísticos ficam localmente em:

- `results/article/summary/metrics_summary.csv`;
- `results/article/summary/precision_assessment.csv`;
- `results/article/summary/pairwise_comparisons.csv`;
- `results/article/raw/attack_runs.csv`;
- `results/article/raw/benign_runs.csv`.

A figura combinada é gerada em `results/article/plots/graficos_combinados.png`.

[Voltar ao índice da documentação](README.md)
