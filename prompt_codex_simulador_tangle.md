# Prompt para o Codex — simulador científico da Tangle

Quero que você atue como engenheiro de software científico e pesquisador de sistemas distribuídos. Inspecione primeiro o repositório e qualquer `AGENTS.md`, artigo, `.bib`, configuração ou código existente. Em seguida, implemente, teste, documente e execute um simulador científico reprodutível da Tangle para comparar três Tip Selection Algorithms (TSAs):

1. Uniform Random Tip Selection (URTS);
2. Markov Chain Monte Carlo / Weighted Random Walk (MCMC/WRW);
3. Sector-Based Weighted Random Walk (SB-WRW), de Ahmed et al., DOI `10.1016/j.iot.2025.101819`.

O objetivo não é produzir números que favoreçam um algoritmo. O objetivo é produzir resultados corretos, auditáveis, reprodutíveis e estatisticamente defensáveis para um artigo científico. Não ajuste parâmetros depois de observar os resultados com a finalidade de obter um ranking esperado. Não hardcode valores de métricas ou gráficos. Resultados negativos, empates e inversões de ranking são válidos se decorrerem do modelo preregistrado.

## 1. Regra de implementação e conformidade do SB-WRW

Antes de programar o SB-WRW, localize e leia o texto completo do artigo:

> Imen Ahmed, Mariem Turki, Mouna Baklouti e Bouthaina Dammak. “SB-WRW: Balancing fairness and security in IoT-adapted tip selection for the tangle”. Internet of Things, v. 34, 101819, 2025. DOI: 10.1016/j.iot.2025.101819.

Crie `docs/sb_wrw_traceability.md` contendo, para cada regra do algoritmo:

- trecho descrito em paráfrase, sem copiar longos trechos protegidos;
- página, seção, algoritmo, figura ou equação de origem;
- parâmetro correspondente no código;
- classe/função e teste que implementam a regra.

O simulador deve conter desde já a especificação operacional completa descrita na Seção 5.3 deste prompt. Ela formaliza os elementos confirmados publicamente para o SB-WRW: divisão geográfica em setores, prioridade por proximidade/localização e transição ponderada pelo peso acumulado do WRW. Não confunda SB-WRW com “Self-Biased WRW”, Age-Aware WRW (AA-WRW), DA-IOTA, G-IOTA ou uma heurística com viés de idade.

Implemente dois identificadores de conformidade no código e nos resultados:

- `sb_wrw_conformance: paper_exact`: permitido somente se todas as regras e parâmetros tiverem sido conferidos no texto completo do artigo e registrados em `docs/sb_wrw_traceability.md`;
- `sb_wrw_conformance: operational_reproduction`: usa exatamente a formalização da Seção 5.3 deste prompt quando o artigo completo não estiver disponível.

No segundo caso, o algoritmo deve aparecer em código como `OperationalSBWRW` e nos artefatos como `SB-WRW (reprodução operacional)`. Ele pode ser executado, testado e comparado, mas o README, a metodologia e as limitações devem informar que a correspondência exata com o pseudocódigo dos autores ainda precisa ser verificada. Não esconda essa distinção.

Quando o PDF completo estiver disponível, compare regra por regra. Se houver divergência, a definição publicada prevalece para o modo `paper_exact`; preserve o modo operacional como variante separada para garantir reprodutibilidade de resultados anteriores.

## 2. Escopo científico e afirmação permitida

Os três casos de uso são perfis sintéticos inspirados em PACTCare, TLIP e ElaadNL. Eles não devem ser apresentados como traces reais dessas plataformas, a menos que o repositório contenha datasets reais e documentação de sua proveniência.

Toda documentação, tabela e legenda deve usar formulação equivalente a:

> “cenários sintéticos inspirados nas características operacionais dos casos de uso”.

Crie `docs/parameter_provenance.md`. Para cada parâmetro, registre uma das origens:

- `source`: valor retirado de artigo/dataset, com referência e página/tabela;
- `derived`: valor calculado a partir de uma fonte, com fórmula;
- `assumption`: hipótese experimental ainda não calibrada;
- `common_control`: valor comum escolhido para isolar o efeito do TSA.

Nunca fabrique citações. Se um valor não tiver fonte, classifique-o como hipótese. Os valores iniciais abaixo são configurações sintéticas de partida, não dados empíricos das três plataformas.

## 3. Tecnologia, estrutura e execução

Use Python 3.12 ou a versão já adotada no repositório. Prefira:

- `numpy` para RNG e cálculo numérico;
- `pandas` e `pyarrow` para resultados tabulares;
- `scipy` para rotinas estatísticas;
- `matplotlib`/`seaborn` para figuras;
- `pydantic` ou dataclasses validadas e YAML para configuração;
- `pytest` e, quando útil, `hypothesis` para testes.

Não use `networkx` no loop principal do simulador. Ele pode ser usado apenas como implementação-oráculo em testes pequenos. Para o motor principal, use estruturas compactas: arrays/listas indexadas por ID inteiro, conjuntos de pontas, listas de pais e aprovadores diretos, vetor de pesos, metadados de setor/tempo/ramo e marcações inteiras para visitas sem realocar grandes conjuntos.

Use um motor de eventos discretos com fila de prioridade (`heapq`) e tempo contínuo. Não avance o relógio em passos fixos. Os eventos mínimos são emissão/seleção, anexação/visibilidade e eventos do ataque. Preserve ordenação determinística para eventos com o mesmo timestamp usando um contador monotônico.

Estrutura sugerida, adaptando-a ao repositório existente:

```text
src/tangle_sim/
  config.py
  cli.py
  domain/
    transaction.py
    tangle.py
    conflicts.py
  algorithms/
    base.py
    urts.py
    mcmc.py
    sb_wrw.py
  simulation/
    events.py
    engine.py
    workloads.py
    attack.py
    rng.py
  metrics/
    confirmation.py
    abandonment.py
    survival.py
    bootstrap.py
  reporting/
    plots.py
    tables.py
configs/
  scenarios.yaml
  pilot.yaml
  paper.yaml
docs/
tests/
results/               # ignorado pelo Git, salvo com manifestos
```

Exponha comandos equivalentes a:

```bash
python -m tangle_sim validate --config configs/paper.yaml
python -m tangle_sim run --config configs/pilot.yaml --output results/pilot
python -m tangle_sim run --config configs/paper.yaml --output results/paper
python -m tangle_sim report --input results/paper --language pt-BR
pytest -q
```

Adicione `Makefile` ou tarefas equivalentes para `test`, `validate`, `pilot`, `paper` e `report`. Fixe versões das dependências. Se o repositório já usa outra ferramenta, integre-se a ela sem duplicar o ecossistema.

## 4. Modelo da Tangle

Implemente uma abstração explícita e documentada da Tangle/IOTA 1.0 usada na literatura dos TSAs, não uma alegação de reprodução da rede IOTA atual.

### 4.1 Transação

Cada transação deve ter pelo menos:

- `id` inteiro e imutável;
- `issue_time`, `attachment_time` e, se modelado, `visible_time`;
- `issuer_id` e `sector_id`;
- um ou dois pais aprovados;
- lista de aprovadores diretos;
- `own_weight`, inicialmente 1 para todos os emissores, salvo experimento preregistrado;
- `cumulative_weight` sem contagem duplicada de descendentes;
- estado honesto/adversarial;
- `conflict_id` e `branch_id` quando aplicável;
- primeira aprovação direta, idade, confirmação e estado de ponta.

As arestas devem seguir uma convenção única e documentada: a nova transação aponta para transações anteriores que ela aprova; a caminhada em direção às pontas percorre os aprovadores no sentido do futuro. Pais precisam ser temporalmente anteriores à transação filha. O grafo deve permanecer acíclico.

### 4.2 Pontas e concorrência

Uma ponta é uma transação sem aprovadores diretos na visão disponível para a seleção. A transação seleciona dois pais distintos e compatíveis sempre que houver pelo menos dois. Durante o bootstrap, permita um único pai (genesis) e documente a exceção; não crie silenciosamente duas arestas iguais.

Modele seleção com visão atrasada: a transação escolhe pais usando um snapshot anterior ao momento de anexação. Assim, duas transações concorrentes podem selecionar as mesmas pontas, que é essencial para o crescimento realista do conjunto de pontas. O atraso total deve ser configurável e amostrado de uma distribuição positiva. Não remova uma ponta retroativamente apenas porque outra transação ainda não visível a selecionou.

Se o artigo do SB-WRW exigir visão por nó/setor e propagação diferenciada, essa regra prevalece e deve ser aplicada igualmente aos algoritmos comparados sempre que representar infraestrutura, não lógica exclusiva do TSA. Caso contrário, use um snapshot global atrasado comum aos três algoritmos.

### 4.3 Peso acumulado

Defina o peso acumulado de `v` como o peso próprio de `v` mais a soma dos pesos próprios de todas as transações que a aprovam direta ou indiretamente, contando cada descendente uma única vez mesmo em estruturas em diamante.

Implemente uma versão lenta de referência para DAGs pequenos e uma versão incremental otimizada. Compare ambas em testes aleatórios pequenos. Nunca conte duas vezes uma transação que alcança o mesmo ancestral pelos dois pais.

## 5. Algoritmos de seleção de pontas

Defina uma interface comum, por exemplo:

```python
class TipSelector(Protocol):
    def select_tip(self, view: TangleView, context: SelectionContext, rng: Generator) -> int: ...
    def select_parents(self, view: TangleView, context: SelectionContext, rng: Generator) -> tuple[int, ...]: ...
```

O mesmo snapshot, regra de compatibilidade, ponto de entrada, atraso, workload e stream de eventos externos deve ser usado pelos três algoritmos. Apenas a política de seleção pode mudar.

### 5.1 URTS

- obtenha todas as pontas visíveis, válidas e compatíveis;
- faça amostragem uniforme sem reposição de duas pontas;
- não use peso, idade ou setor na probabilidade;
- se a primeira ponta determinar um ramo de conflito, restrinja a segunda a pontas compatíveis e registre retries/fallbacks;
- teste uniformidade em um conjunto controlado usando teste qui-quadrado com tolerância não frágil.

### 5.2 MCMC/WRW

Implemente duas caminhadas independentes, uma para cada ponta. O ponto de entrada/depth deve ser definido pela referência e configurado, nunca escolhido de forma diferente por algoritmo para melhorar resultado.

Em uma transação atual `x`, os candidatos `y` são os aprovadores diretos visíveis no sentido das pontas. A probabilidade deve seguir a formulação da literatura, tipicamente:

```text
P(x -> y) = exp[-alpha * (CW(x) - CW(y))]
            / sum_z exp[-alpha * (CW(x) - CW(z))]
```

Confirme a orientação e a forma exata na fonte adotada e cite-a em `docs/methodology.md`. Calcule com `logsumexp` ou softmax estabilizado. Não use exponencial ingênua sujeita a overflow/underflow.

Requisitos:

- `alpha = 0` deve resultar em transição uniforme entre os aprovadores diretos elegíveis;
- alpha maior deve favorecer o ramo de maior peso acumulado;
- a caminhada termina apenas em ponta válida/visível ou em condição de fallback explicitamente registrada;
- imponha limite defensivo de passos; excedê-lo deve gerar diagnóstico, não resultado silencioso;
- a escolha de duas pontas deve ser independente, seguida de regra comum de distinção/compatibilidade;
- alpha e o ponto de entrada devem ser preregistrados. Faça análise de sensibilidade separada, sem escolher retrospectivamente o alpha “vencedor”.

### 5.3 SB-WRW

Implemente o SB-WRW sobre a mesma caminhada, peso acumulado, ponto de entrada, alpha, snapshot e validação de conflitos usados pelo MCMC. A única diferença primária deve ser o filtro geográfico local-first definido abaixo. Isso impede que diferenças de infraestrutura sejam atribuídas indevidamente ao algoritmo.

#### 5.3.1 Representação espacial e setores

Cada emissor `i` possui coordenadas fixas `pos(i) = (x_i, y_i)` e um setor `s(i)`. Cada transação `v` herda o setor e as coordenadas de seu emissor:

```text
s(v)   = s(issuer(v))
pos(v) = pos(issuer(v))
```

Use uma região retangular normalizada `[0,1] x [0,1]`, salvo existência de coordenadas reais. Particione-a deterministicamente em uma grade regular:

- PACTCare: `2 x 3 = 6` setores;
- TLIP: `3 x 4 = 12` setores;
- ElaadNL: `4 x 5 = 20` setores.

Numere os setores por linha e coluna. A distância setorial primária é a distância de Manhattan entre células:

```text
D_s(a,b) = |row(a)-row(b)| + |col(a)-col(b)|
```

Assim, `D_s=0` significa mesmo setor, `D_s=1` significa setor adjacente e valores maiores representam anéis sucessivamente mais distantes. Guarde também a distância euclidiana entre coordenadas para diagnósticos, mas não a use na decisão primária sem alterar a versão da configuração.

A geração das coordenadas deve ser parte do workload comum: para uma réplica e cenário, MCMC, URTS e SB-WRW recebem exatamente os mesmos emissores, posições e setores. As coordenadas permanecem fixas durante a réplica. Não permita que o SB-WRW reposicione nós para melhorar resultados.

#### 5.3.2 Contexto da caminhada

Para uma nova transação emitida pelo nó `i`, defina `s_i = s(i)`. As duas caminhadas usadas para escolher os pais utilizam `s_i` como setor-alvo. O setor-alvo não muda ao longo da caminhada.

Escolha o ponto de entrada `x_0` pela mesma regra do MCMC. Não escolha um ponto de entrada local especial para o SB-WRW, a menos que o artigo completo determine isso no modo `paper_exact`.

Em cada estado `x`, obtenha primeiro o conjunto de aprovadores diretos visíveis, válidos e compatíveis:

```text
C(x) = {y : y aprova diretamente x, y está visível e y é válido/compatível}
```

Se `C(x)` for vazio, `x` é a ponta alcançada e a caminhada termina.

#### 5.3.3 Filtro setorial local-first

Particione os candidatos pelo anel geográfico em relação ao emissor da nova transação:

```text
C_r(x, i) = {y em C(x) : D_s(s_i, s(y)) = r}
```

Encontre o menor anel que contém ao menos um candidato:

```text
r*(x,i) = min {r >= 0 : C_r(x,i) não é vazio}
```

O conjunto elegível para o passo é exclusivamente:

```text
E_SB(x,i) = C_r*(x,i)
```

Consequências obrigatórias:

1. se houver aprovador direto no mesmo setor (`r*=0`), nenhum aprovador de outro setor pode ser escolhido naquele passo;
2. se não houver candidato local, expanda para o anel mais próximo com candidatos;
3. nunca encerre a caminhada em uma transação que ainda possui aprovadores visíveis apenas porque não existem aprovadores locais;
4. nunca salte diretamente para uma ponta arbitrária de outro setor;
5. a expansão é feita novamente em cada passo, pois o conjunto de aprovadores muda ao longo do caminho.

Esse filtro representa a priorização setorial. Ele não contém fator de idade, bônus por ser ponta ou penalidade por já ter sido selecionada.

#### 5.3.4 Transição ponderada dentro do setor escolhido

Dentro de `E_SB(x,i)`, aplique exatamente a regra de peso acumulado do WRW:

```text
logit(y) = -alpha * (CW(x) - CW(y))

P_SB(x -> y | i) = exp(logit(y))
                     / sum_{z em E_SB(x,i)} exp(logit(z))
```

Como `CW(x)` é constante entre candidatos no mesmo passo, uma forma numericamente equivalente é:

```text
P_SB(x -> y | i) = softmax(alpha * CW(y))
```

Use softmax estabilizado por subtração do maior logit. Não misture peso acumulado com distância em unidades incompatíveis numa soma arbitrária. Neste modelo, distância define o conjunto elegível e o peso acumulado define a probabilidade dentro dele.

Propriedades esperadas:

- com `alpha=0`, a escolha é uniforme entre candidatos do anel elegível mais próximo;
- com alpha maior, candidatos de maior peso acumulado dentro desse anel recebem maior probabilidade;
- com apenas um setor, o SB-WRW deve ser probabilisticamente equivalente ao MCMC/WRW;
- um candidato remoto de peso muito alto não pode superar um candidato local enquanto existir candidato no anel local; essa é a diferença deliberada para o WRW.

O alpha primário deve ser o mesmo do MCMC para que a comparação isole o componente setorial. Qualquer alpha diferente deve ser somente análise de sensibilidade ou exigência documentada do artigo completo.

#### 5.3.5 Seleção das duas pontas

Execute duas caminhadas independentes a partir do mesmo tipo de ponto de entrada, usando substreams RNG independentes e o mesmo setor-alvo `s_i`.

Procedimento:

1. execute `walk_1` e obtenha `tip_1`;
2. execute `walk_2`, excluindo `tip_1` apenas no endpoint e respeitando o mesmo filtro setorial em todos os passos;
3. se `walk_2` terminar em `tip_1`, reinicie a segunda caminhada com novo substream, até `max_tip_retries`;
4. se não for possível obter uma segunda ponta distinta e compatível, use o fallback comum aos demais algoritmos: escolha, sem reposição, entre pontas visíveis compatíveis do anel setorial mais próximo; dentro do anel use pesos de endpoint derivados do WRW;
5. se existir somente uma ponta compatível em todo o snapshot, retorne um único pai e registre `single_parent_bootstrap_or_degenerate`; nunca duplique a mesma aresta.

O segundo pai deve ser compatível com o ramo de conflito determinado pelo primeiro. Se o primeiro endpoint pertencer a um ramo de gasto duplo, a segunda caminhada elimina candidatos incompatíveis antes do cálculo de `r*` e das probabilidades.

#### 5.3.6 Pseudocódigo normativo

Implemente comportamento equivalente a:

```python
def sb_wrw_walk(view, issuer_sector, entry_point, alpha, rng, excluded_tips=frozenset()):
    x = entry_point

    for step in range(MAX_WALK_STEPS):
        candidates = [
            y for y in view.direct_approvers(x)
            if view.is_visible(y)
            and view.is_valid(y)
            and view.is_branch_compatible(y)
        ]

        if not candidates:
            if x in excluded_tips:
                raise ExcludedEndpointReached(x)
            if not view.is_tip(x):
                raise InconsistentViewError(x)
            return x

        nearest_ring = min(
            sector_distance(issuer_sector, view.sector(y))
            for y in candidates
        )
        eligible = [
            y for y in candidates
            if sector_distance(issuer_sector, view.sector(y)) == nearest_ring
        ]

        logits = [alpha * view.cumulative_weight(y) for y in eligible]
        probabilities = stable_softmax(logits)
        x = rng.choice(eligible, p=probabilities)

    raise WalkStepLimitExceeded(entry_point)
```

A implementação real pode ser otimizada, mas deve produzir a mesma distribuição do pseudocódigo em grafos pequenos.

#### 5.3.7 Configuração

Inclua configuração explícita:

```yaml
sb_wrw:
  conformance: operational_reproduction  # mudar para paper_exact só após rastreabilidade
  sector_assignment: regular_grid
  sector_distance: manhattan
  locality_policy: nearest_nonempty_ring
  alpha: ${algorithms.mcmc.alpha}
  max_walk_steps: 10000
  max_tip_retries: 32
  endpoint_fallback: nearest_ring_weighted_without_replacement
  age_bias: disabled
  continuous_distance_bias: disabled
```

Valide a configuração e rejeite combinações desconhecidas. O `manifest.json` deve salvar essa configuração resolvida e o identificador de conformidade.

#### 5.3.8 Diagnósticos obrigatórios

Por réplica, exporte:

- número e percentual de passos com `r*=0`, `r*=1`, `r*=2` e `r*>=3`;
- distância setorial média e máxima dos passos escolhidos;
- distância euclidiana média entre emissor e endpoints;
- comprimento médio, mediana, p95 e máximo das caminhadas;
- retries para segunda ponta;
- fallbacks e seleções com pai único;
- distribuição de endpoints por setor;
- percentual de seleções em que o SB-WRW divergiu do candidato que teria maior probabilidade no MCMC sem filtro local.

Esses diagnósticos são necessários para demonstrar que o componente setorial realmente foi exercitado. Uma execução em que quase todos os passos possuem apenas um candidato não valida o comportamento do SB-WRW.

#### 5.3.9 Testes específicos

Crie pelo menos os seguintes testes:

1. `same_sector_has_absolute_priority`: candidato local é sempre elegível e candidato remoto é excluído, independentemente do peso remoto;
2. `expands_to_nearest_nonempty_ring`: sem candidato local, usa somente o menor anel disponível;
3. `wrw_weighting_inside_ring`: dois candidatos no mesmo anel seguem a proporção prevista pela fórmula;
4. `alpha_zero_is_uniform_inside_ring`;
5. `one_sector_matches_mcmc_distribution`;
6. `remote_only_path_does_not_stop_early`: a caminhada expande em vez de declarar falsa ponta;
7. `second_tip_is_distinct_and_compatible`;
8. `diamond_does_not_double_count_cumulative_weight`;
9. `stable_softmax_with_large_weights`;
10. `optimized_implementation_matches_normative_reference` em milhares de caminhadas sobre DAGs pequenos;
11. teste estatístico em que as frequências observadas ficam dentro de tolerância da distribuição teórica;
12. teste de snapshot atrasado garantindo que o algoritmo não enxerga aprovadores futuros.

Não acrescente viés de idade, taxa de seleção anterior, bônus direto para ponta abandonada ou ajuste dinâmico de alpha a esta definição. Qualquer uma dessas mudanças cria outro algoritmo e exige nome, configuração e resultados separados.

## 6. Cenários sintéticos

Implemente os cenários via YAML, sem `if scenario == ...` espalhado pelo código. Todos os valores devem ser sobrescrevíveis por CLI/configuração.

Use como ponto de partida, marcado como `assumption`, o seguinte desenho:

| Cenário | Emissores | Setores | Processo de chegada | Carga-alvo aproximada | Atraso mediano inicial | Característica representada |
|---|---:|---:|---|---:|---:|---|
| PACTCare | 60 | 6 | Poisson homogêneo | 4 tx/s | 0,25 s de rede + atraso comum de processamento | carga moderada, instituições delimitadas e criticidade |
| TLIP | 240 | 12 | Poisson modulado por Markov, 8 tx/s no estado baixo e 24 tx/s no alto | média de 12 tx/s | 0,60 s de rede + atraso comum | múltiplos agentes, heterogeneidade e rajadas operacionais |
| ElaadNL | 1000 | 20 | tráfego periódico com jitter mais fundo Poisson | média próxima de 40 tx/s | 0,10 s de rede + atraso comum | alto volume M2M e emissões temporalmente próximas |

Para o MMPP do TLIP, use inicialmente duração média de 45 s no estado baixo e 15 s no estado alto. Para ElaadNL, use inicialmente fundo de 20 tx/s e rajadas de 120 tx/s por 5 s a cada 30 s, com jitter configurável; documente que a média resultante é uma hipótese sintética. Ajuste apenas se uma fonte ou dataset justificar outros números.

Distribua emissores entre setores de forma fixa por réplica. PACTCare pode iniciar balanceado; TLIP deve permitir heterogeneidade configurável entre organizações; ElaadNL deve representar clusters locais de dispositivos. O mesmo mapa de emissores e a mesma sequência exógena devem alimentar todos os TSAs na réplica correspondente.

Configuração temporal inicial:

```yaml
warmup_seconds: 60
measurement_seconds: 300
drain_seconds: 120
abandonment_age_seconds: 120
```

- gere transações durante warm-up e medição;
- calcule as métricas principais apenas para transações honestas emitidas na janela de medição;
- permita confirmações durante o drain;
- não inclua transações jovens sem tempo de maturação no denominador de abandono;
- faça uma análise de sensibilidade em pelo menos carga (`0.5x`, `1x`, `2x`), atraso e alpha, salva separadamente dos resultados primários.

## 7. Separação entre experimento benigno e experimento de segurança

Execute duas baterias independentes:

### Bateria A — operação sem ataque

Usada exclusivamente para:

1. taxa de confirmação;
2. percentual de pontas honestas deixadas para trás.

### Bateria B — ataque de gasto duplo/parasite chain

Usada para:

3. tempo até consolidação da fraude;
4. taxa de sucesso da fraude, resultado auxiliar obrigatório;
5. curva de sobrevivência, resultado auxiliar obrigatório.

Não calcule confirmação e abandono principais em uma execução contaminada pelo ataque. Não misture execuções benignas e adversariais na mesma média.

## 8. Modelo de fraude

Implemente um ataque de parasite chain/gasto duplo conforme a literatura escolhida, registrando a fonte e as diferenças de abstração.

Modelo mínimo:

1. após o warm-up, crie duas transações conflitantes que gastam a mesma entrada e possuem um ancestral comum: a transação honesta `H0`, publicada, e a fraudulenta `F0`, inicialmente privada;
2. a rede honesta continua emitindo no ramo de `H0`;
3. o atacante constrói em privado uma subtangle/parasite chain sobre `F0` com taxa proporcional à taxa honesta;
4. após atraso de liberação preregistrado, publique o ramo fraudulento;
5. depois da publicação, o TSA e a regra de compatibilidade determinam a evolução, sem forçar artificialmente honestos a escolher o atacante;
6. pais pertencentes a ramos conflitantes não podem ser aprovados conjuntamente; retries e decisões de ramo devem ser explícitos.

Use inicialmente, como hipóteses comuns aos cenários:

```yaml
attacker_rate_fraction: 0.30
private_build_seconds: 30
attack_observation_seconds: 300
fraud_confidence_threshold: 0.95
confidence_stability_seconds: 10
confidence_check_interval_seconds: 2
```

Inclua análise de sensibilidade para fração adversarial, por exemplo `0.10`, `0.20`, `0.30` e `0.40`. Os resultados primários devem usar o valor preregistrado antes da execução final.

Defina confirmação/confiança de conflito de forma coerente com a Tangle: estime a probabilidade de uma ponta obtida pelo TSA aprovar direta ou indiretamente `H0` ou `F0`. Use amostras Monte Carlo independentes do RNG de evolução, ou cálculo exato quando viável. Uma fraude consolida quando a confiança de `F0` alcança `0.95` e permanece acima do limiar durante a janela de estabilidade. Documente também se foi exigida queda correspondente da confiança de `H0`.

Se a definição adotada do artigo-base diferir, implemente a fonte e explique a mudança. Nunca defina “fraude consolidada” apenas como “o ramo atingiu N transações” sem justificativa, pois isso ignora a probabilidade de seleção pelo TSA.

## 9. Definições das métricas

As fórmulas abaixo devem aparecer em `docs/methodology.md` e nos metadados dos resultados.

### 9.1 Confiança de confirmação

Para uma transação `v` e snapshot `G(t)`, defina:

```text
CC_v(t) = Pr[TSA termina em uma ponta que aprova v direta ou indiretamente]
```

Para URTS, calcule exatamente pela proporção de pontas válidas que aprovam `v` quando possível. Para caminhadas, estime por amostragem independente e registre número de amostras e erro Monte Carlo. Use inicialmente `K = 5.000` caminhadas por snapshot final, podendo agregar frequências por ponta para não repetir travessias de ancestrais. Use stream RNG exclusivo de medição.

### 9.2 Taxa de confirmação (TC)

Considere o conjunto `E_r` de transações honestas elegíveis emitidas na janela de medição da réplica `r` e maduras até o fim do drain:

```text
TC_r = 100 * |{v em E_r : CC_v(t_final) >= 0.95}| / |E_r|
```

Registre numerador, denominador e percentual. Não arredonde antes da agregação.

### 9.3 Percentual de pontas honestas deixadas para trás

Uma transação honesta é “deixada para trás” se, no fim do drain:

- continua sendo uma ponta na visão global consolidada;
- sua idade é pelo menos `abandonment_age_seconds`;
- pertence à janela elegível.

Defina:

```text
P_ab,r = 100 * N_abandonadas,r / |E_r|
```

Também exporte `N_abandonadas` e a distribuição de idade das pontas como diagnóstico. Não troque silenciosamente o denominador por “número de pontas finais”. Se o artigo já preregistrou outro denominador, implemente ambos, identifique claramente qual é o principal e nunca compare percentuais de definições diferentes.

### 9.4 Tempo até consolidação da fraude

O tempo de origem é a publicação de `F0`, não o início secreto da construção, salvo definição diferente explicitamente reportada. Se a fraude não consolidar até o horizonte, a observação é censurada à direita.

Não calcule como métrica principal a média apenas entre ataques bem-sucedidos. Isso introduz viés de seleção. Use:

- curva de Kaplan–Meier para tempo até fraude;
- Restricted Mean Survival Time (RMST) até o horizonte comum `tau` como “tempo médio restrito até consolidação da fraude”;
- intervalo de confiança de 95% por bootstrap de réplicas;
- taxa de sucesso até `tau` como resultado auxiliar ao lado do RMST.

No CSV, preserve `event_observed`, `time_to_event_or_censoring` e `censoring_horizon`. Para facilitar o texto do artigo, o gráfico principal pode ter o título “Tempo médio até a consolidação de fraude (s)”, mas legenda e metodologia devem declarar que a barra representa RMST e que valores maiores indicam maior atraso/maior resistência. Se todas as observações forem censuradas, reporte RMST igual ao horizonte com a taxa de sucesso zero; não escreva “tempo infinito”.

## 10. Desenho experimental e aleatoriedade

Use a réplica completa como unidade experimental. Não trate transações individuais da mesma execução como réplicas independentes.

Use `numpy.random.SeedSequence` com uma seed raiz configurável. Separe streams para:

- chegadas/workload;
- emissores e setores;
- atrasos de rede/processamento;
- tip selection de cada algoritmo;
- ataque;
- estimação de confiança;
- bootstrap.

Empregue Common Random Numbers: para a mesma combinação `(cenário, réplica)`, preregistre e reutilize entre os três algoritmos exatamente os mesmos tempos de chegada, emissores, setores, atrasos exógenos e agenda adversarial. Como os algoritmos consomem quantidades diferentes de aleatoriedade, use streams próprios para seleção e não dependa da ordem global de chamadas RNG.

Use:

- piloto com 10 réplicas apenas para depuração, tempo e estimativa de variância;
- execução primária com mínimo de 30 e alvo de 100 réplicas por `(cenário, algoritmo, bateria)`, definido por análise de poder/precisão baseada no piloto;
- a mesma quantidade de réplicas para os três algoritmos;
- bootstrap pareado por índice de réplica, aproveitando Common Random Numbers;
- intervalos de confiança de 95% para todas as barras;
- comparações pareadas entre algoritmos por cenário, com tamanho de efeito, IC e correção de Holm para múltiplas comparações.

Não use apenas p-valores. Exporte diferenças absolutas, relativas e intervalos de confiança. Registre previamente em um manifesto o número final de réplicas e os parâmetros antes da execução final. Não altere o YAML primário depois de olhar `results/paper`; alterações devem criar uma nova configuração/versionamento.

## 11. Resultados e formatos

Salve, no mínimo:

```text
results/<run_id>/
  manifest.json
  config_resolved.yaml
  environment.txt
  raw/
    benign_runs.csv
    attack_runs.csv
    benign_runs.parquet
    attack_runs.parquet
  summary/
    metrics_summary.csv
    pairwise_comparisons.csv
    sensitivity_summary.csv
  plots/
    taxa_confirmacao.png
    taxa_confirmacao.pdf
    taxa_confirmacao.svg
    pontas_abandonadas.png
    pontas_abandonadas.pdf
    pontas_abandonadas.svg
    tempo_fraude.png
    tempo_fraude.pdf
    tempo_fraude.svg
    metricas_combinadas.png
    curvas_sobrevivencia.pdf
```

Cada linha bruta deve incluir cenário, algoritmo, réplica, seeds/identificadores de stream, configuração relevante, contagens, denominadores, métrica não arredondada, duração, versão do simulador e diagnósticos (retries, fallback, passos de caminhada e seleções locais do SB-WRW).

O `manifest.json` deve incluir:

- timestamp UTC;
- hash do Git ou indicação de worktree sem commit;
- hash SHA-256 da configuração resolvida e do artigo/PDF usado no modo `paper_exact`; no modo operacional, hash/versão da especificação normativa da Seção 5.3;
- versões de Python e dependências;
- sistema operacional;
- seed raiz;
- número de réplicas;
- comandos executados;
- definição textual e versão de cada métrica.

## 12. Três gráficos de barras obrigatórios

Gere três gráficos de barras agrupadas, todos a partir de `metrics_summary.csv`, sem números codificados na função de plotting.

Em cada gráfico:

- eixo X: `PACTCare`, `TLIP`, `ElaadNL`, nesta ordem;
- três barras por cenário: `URTS`, `MCMC`, `SB-WRW`;
- cores fixas e iguais em todas as figuras:
  - URTS: azul `#1f77b4`;
  - MCMC: laranja `#ff7f0e`;
  - SB-WRW: verde `#2ca02c`;
- barras iniciando em zero, sem eixo truncado enganoso;
- barras de erro com IC de 95%;
- rótulo numérico sobre cada barra com precisão adequada;
- legenda horizontal centralizada no topo ou em posição que não cubra dados;
- fonte legível no formato de duas colunas do artigo;
- fundo branco, grade horizontal discreta, sem 3D;
- exportação PNG a pelo menos 300 dpi e também PDF/SVG vetorial;
- títulos e eixos em português correto.

Se a conformidade for `operational_reproduction`, use `SB-WRW*` na legenda e acrescente na caption: `* reprodução operacional local-first; correspondência com o pseudocódigo integral dos autores ainda não verificada`. Somente use a legenda simples `SB-WRW` quando `sb_wrw_conformance=paper_exact`.

Figuras:

1. `Taxa de confirmação` — eixo Y `Taxa de confirmação (%)`, preferencialmente 0–100. Como diferenças próximas de 100% ficam visualmente pequenas, gere adicionalmente uma figura suplementar ampliada ou painel de diferenças, mas não substitua nem trunque a figura principal.
2. `Percentual médio de pontas honestas deixadas para trás` — eixo Y `Pontas honestas deixadas para trás (%)`.
3. `Tempo médio até a consolidação de fraude` — eixo Y `Tempo até a consolidação de fraude (s)`; a nota/caption deve dizer que é RMST até `tau`, com censura à direita, e que maior é melhor do ponto de vista de resistência.

Também produza uma figura combinada com três painéis, mantendo os três arquivos individuais como entregáveis principais.

## 13. Validação e testes obrigatórios

Crie testes unitários, de propriedade, integração e regressão. No mínimo:

### DAG e pesos

- genesis e bootstrap;
- pais sempre anteriores;
- ausência de ciclos;
- atualização correta do conjunto de pontas;
- diamante com descendente contado uma única vez no peso acumulado;
- equivalência do peso incremental com o algoritmo lento de referência em DAGs pequenos aleatórios;
- conflitos incompatíveis nunca aparecem como pais da mesma transação.

### TSAs

- URTS aproximadamente uniforme;
- MCMC com `alpha=0` uniforme entre aprovadores diretos;
- MCMC com alpha alto favorece deterministicamente, em frequência estatística, o candidato de maior peso;
- probabilidades finitas e soma igual a 1 mesmo com pesos grandes;
- duas caminhadas usam streams independentes;
- limite de passos e fallback são observáveis;
- SB-WRW reproduz propriedades/exemplos do artigo, incluindo setor, distância e fallback.

### Workloads e RNG

- mesma seed produz bytes/CSV equivalentes, exceto campos explicitamente voláteis;
- Common Random Numbers gera a mesma sequência exógena para os três TSAs;
- taxas empíricas de Poisson/MMPP/periódico ficam dentro de tolerância estatística;
- nenhuma seed é reutilizada acidentalmente entre evolução e medição.

### Métricas

- TC e abandono em DAGs manuais;
- exclusão correta de warm-up e transações imaturas;
- confiança por URTS comparada ao valor exato;
- Kaplan–Meier e RMST comparados com exemplos manuais ou biblioteca confiável;
- caso sem fraudes, todas censuradas;
- caso com todas as fraudes imediatas;
- bootstrap determinístico com seed.

### Validação científica

- em um cenário Poisson/URTS controlado, compare o número estacionário de pontas com a previsão analítica aplicável da literatura (por exemplo, relação com taxa de chegada e atraso); reporte erro e condições de validade, sem transformar isso em teste frágil de igualdade exata;
- compare o motor otimizado com uma implementação simples em grafos pequenos;
- faça sanity checks de monotonicidade: aumentar o drain não pode desconfirmar transações; aumentar a idade mínima não pode aumentar o conjunto elegível de abandonadas; peso acumulado nunca diminui;
- execute perfil de desempenho e garanta que otimizações não mudem métricas.

## 14. Documentação científica

Crie:

- `README.md`: instalação, comandos, estrutura e reprodução completa;
- `docs/methodology.md`: modelo, equações, métricas, censura, ataque e estatística;
- `docs/sb_wrw_traceability.md`: rastreabilidade do artigo para o código;
- `docs/parameter_provenance.md`: fonte/derivação/hipótese de cada parâmetro;
- `docs/validation.md`: testes analíticos, implementação de referência e resultados de validação;
- `docs/limitations.md`: ameaças à validade interna, externa, de construção e de conclusão;
- `docs/article_results_template.md`: tabela e texto-modelo que descrevem resultados sem afirmar causalidade além do experimento.

Use como referências técnicas mínimas, além das referências já existentes no repositório:

- Ahmed et al. (2025), SB-WRW, DOI: <https://doi.org/10.1016/j.iot.2025.101819>;
- Kusmierz et al., “Properties of the Tangle for Uniform Random and Random Walk Tip Selection”: <https://arxiv.org/abs/2001.07734> e versão publicada na IEEE;
- Popov, “The Tangle”, para a definição histórica de peso, confiança e parasite chain, deixando claro que é documento técnico histórico;
- Penzkofer et al., “Parasite Chain Detection in the IOTA Protocol”: <https://doi.org/10.4230/OASIcs.Tokenomics.2020.8>;
- Ferraro et al., “On the stability of unverified transactions in a DAG-based Distributed Ledger”: <https://arxiv.org/abs/1901.07302>.

Priorize a versão revisada por pares quando houver. As fontes acima orientam a implementação, mas não autorizam preencher lacunas do SB-WRW com regras de outro algoritmo.

As limitações devem mencionar explicitamente:

- cenários sintéticos, não traces reais;
- abstração da Tangle/IOTA 1.0;
- sensibilidade a alpha, atrasos, taxa adversarial, limiar de confiança e definição de abandono;
- possível erro Monte Carlo na confiança;
- diferença entre segurança simulada e segurança de uma rede implantada;
- SB-WRW dependente da modelagem espacial e da correspondência fiel ao artigo.

## 15. Desempenho e paralelismo

Paralelize por réplica/processo, nunca compartilhando RNG mutável entre workers. A saída deve ser determinística independentemente do número de workers e da ordem de conclusão. Grave arquivos temporários por réplica e faça redução ordenada ao final para evitar corrupção.

Antes da execução primária:

- rode profiler no piloto;
- remova travessias quadráticas desnecessárias;
- mantenha uma implementação clara de referência nos testes;
- estime tempo e memória da execução final;
- se 100 réplicas forem inviáveis no ambiente, calcule a quantidade necessária para o IC-alvo, execute pelo menos 30 quando possível e documente a limitação. Não apresente o piloto de 10 réplicas como resultado final.

## 16. Ordem de trabalho obrigatória

1. Inspecione o repositório e resuma o que já existe.
2. Localize as referências, especialmente o PDF completo do SB-WRW.
3. Escreva um plano curto e identifique decisões científicas ainda ausentes.
4. Crie a documentação de rastreabilidade e a configuração preregistrada antes de observar resultados finais.
5. Implemente o domínio e o motor de eventos.
6. Implemente URTS e MCMC, com testes.
7. Implemente a reprodução operacional normativa da Seção 5.3; se o texto completo estiver disponível, faça em seguida a verificação e, quando aplicável, o modo `paper_exact`.
8. Implemente workloads, ataque, métricas, censura e estatística.
9. Rode todos os testes e validações analíticas.
10. Rode o piloto, corrija defeitos e faça profiling.
11. Congele `configs/paper.yaml` e gere seu hash.
12. Rode a execução primária.
13. Gere CSVs, manifestos, comparações, três gráficos e curvas auxiliares.
14. Revise automaticamente invariantes e procure NaN, denominador zero, sementes duplicadas, censura incorreta e gráficos divergentes do CSV.
15. Entregue um resumo do que foi feito, comandos executados, testes, caminhos dos artefatos, limitações e qualquer bloqueio. Não declare “validado cientificamente” apenas porque os testes passaram; diga exatamente o que foi validado.

## 17. Critérios de aceite

Considere a tarefa concluída somente quando:

- os três TSAs estiverem implementados; o SB-WRW deve estar em modo `paper_exact` rastreado ao artigo ou em modo `operational_reproduction` integralmente rastreado à Seção 5.3 e identificado sem ambiguidade;
- todos os testes passarem;
- o motor otimizado concordar com a referência em casos pequenos;
- workloads idênticos forem reutilizados entre algoritmos;
- métricas tiverem numeradores, denominadores e definições exportados;
- fraude censurada for tratada por Kaplan–Meier/RMST;
- pelo menos um piloto completo tiver sido executado;
- a execução científica final tiver sido executada ou houver bloqueio objetivo documentado;
- os três gráficos separados e a figura combinada tiverem sido gerados diretamente dos CSVs;
- resultados forem reproduzíveis pela seed e configuração resolvida;
- nenhuma conclusão tiver sido escrita com números inventados ou arredondados prematuramente;
- as limitações deixarem claro que PACTCare, TLIP e ElaadNL são perfis sintéticos inspirados nos casos de uso, salvo existência de datasets reais.

Comece agora inspecionando o repositório. Não peça confirmação para escolhas reversíveis de engenharia que já estejam especificadas acima. Pare e peça o PDF/definição somente se isso for indispensável para não falsificar a implementação do SB-WRW.
