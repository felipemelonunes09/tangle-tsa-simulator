# Metodologia executável

Este documento descreve o protocolo realmente codificado em [`configs/article.yaml`](../configs/article.yaml). Em caso de diferença com texto narrativo externo, a configuração resolvida, o manifesto e os dados brutos são a fonte executável de verdade.

## Escopo do modelo

O simulador representa uma abstração histórica da Tangle/IOT como um DAG. Cada nova transação aprova até duas pontas visíveis e compatíveis. Atrasos de anexação criam visões temporais do grafo; conflitos honestos e fraudulentos são mantidos em ramos compatíveis.

Os três perfis são sintéticos:

| Cenário | Emissores | Grade | Workload | Atraso lognormal | Distribuição setorial |
|---|---:|---:|---|---|---|
| PACTCare | 60 | 2 × 3 | Poisson, taxa 4/s | mediana 0,60 s; sigma 0,35 | balanceada |
| TLIP | 240 | 3 × 4 | MMPP, 8/s ou 24/s | mediana 0,60 s; sigma 0,45 | heterogênea |
| ElaadNL | 1000 | 4 × 5 | fundo 20/s e rajadas 120/s | mediana 0,50 s; sigma 0,30 | agrupada |

No TLIP, os estados do MMPP duram em média 45 s no regime baixo e 15 s no regime alto. No ElaadNL, as rajadas duram 5 s a cada 30 s, com jitter de 0,10 s.

## Algoritmos

### URTS

Seleciona uniformemente pontas visíveis e compatíveis. A segunda ponta é escolhida sem repetir a primeira e respeita o ramo da primeira seleção.

### MCMC/WRW

Parte da gênese e caminha pelos aprovadores diretos. Para candidato `y` a partir de `x`, a probabilidade é proporcional a:

```text
exp[-alpha * (CW(x) - CW(y))]
```

Como `CW(x)` é constante entre candidatos do mesmo passo, a implementação usa softmax numericamente estável e é equivalente a `softmax(alpha * CW(y))`.

### SB-WRW 

Usa a mesma transição ponderada do MCMC depois de aplicar um filtro `local-first`:

1. calcula a distância Manhattan entre o setor do emissor e o setor de cada aprovador direto;
2. mantém apenas candidatos no anel não vazio mais próximo;
3. sorteia dentro desse conjunto pelo peso acumulado;
4. repete até alcançar uma ponta.

A idade e um viés contínuo de distância ficam desabilitados. A implementação é identificada como **SB-WRW (reprodução operacional)**, não como reprodução exata do artigo de Ahmed et al. Consulte a [rastreabilidade](sb_wrw_traceability.md).

O mesmo `alpha` é usado por MCMC e SB-WRW em cada cenário:

| Cenário | `alpha` |
|---|---:|
| PACTCare | 2,25 |
| TLIP | 0,1 |
| ElaadNL | 0,1 |

## Bateria benigna

Cada réplica possui:

- aquecimento: 5 s;
- medição: 25 s;
- drenagem quiescente: 5 s;
- idade mínima para abandono: 5 s.

Na drenagem, nenhuma transação nova é emitida; apenas anexações pendentes terminam. A população elegível contém transações honestas emitidas durante a janela de medição e visíveis no snapshot final.

As métricas principais são:

- **taxa de aprovação:** percentual de transações elegíveis com ao menos uma aprovação direta até o snapshot final;
- **pontas abandonadas:** percentual de transações elegíveis que ainda são pontas globais e têm idade mínima de 5 s.

A confiança condicionada ao TSA também é calculada exatamente como diagnóstico, mas não substitui a definição principal de aprovação direta.

## Bateria adversarial

O atacante começa com participação de 36% na emissão total, passa para 70% após 60 s e para 85% após 75 s. A construção privada dura 5 s no PACTCare e 15 s no TLIP e no ElaadNL. Após a publicação de `F0`, a observação dura no máximo 110 s.

Uma consolidação de fraude ocorre quando:

- a confiança exata de `F0` é pelo menos 0,95;
- a confiança do conflito honesto é no máximo 0,05;
- as duas condições permanecem válidas por 2 s consecutivos.

A confiança é verificada a cada 0,5 s. Execuções sem evento até 110 s são censuradas à direita.

## RMST e inferência

O tempo de resistência principal é a média restrita de sobrevivência (RMST) até `tau=110 s`, estimada pela curva de Kaplan–Meier. Valores maiores significam que a consolidação fraudulenta foi retardada por mais tempo dentro do horizonte observado; não significam segurança absoluta.

Cada célula possui dez réplicas. O resumo reporta média, variância amostral com correção de Bessel, desvio-padrão, erro-padrão, erro-padrão relativo e intervalo percentil de bootstrap com 10.000 reamostragens e 95% de confiança. Comparações usam o pareamento por cenário e réplica.

O critério de precisão é:

- margem conservadora de até 5 pontos percentuais para aprovação e abandono;
- margem conservadora de até 5% da média para RMST.

A margem conservadora é a maior distância entre a estimativa e um dos extremos do intervalo. O nível de confiança de 95% não implica automaticamente margem de 5%.

[Voltar ao índice da documentação](README.md)
