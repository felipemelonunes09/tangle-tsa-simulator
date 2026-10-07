PYTHON ?= python
export PYTHONPATH := src
export MPLCONFIGDIR := /tmp/tangle-sim-matplotlib

ARTICLE_CONFIG ?= configs/article.yaml
ARTICLE_OUTPUT ?= results/article
ARTICLE_REFERENCE ?= $(ARTICLE_OUTPUT)

.PHONY: test validate article article-report article-audit article-check paper report
test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) -m tangle_sim validate --config $(ARTICLE_CONFIG)

article:
	$(PYTHON) -m tangle_sim run --config $(ARTICLE_CONFIG) --output $(ARTICLE_OUTPUT)
	$(PYTHON) scripts/assess_precision.py --input $(ARTICLE_OUTPUT)/summary/metrics_summary.csv --attack-runs $(ARTICLE_OUTPUT)/raw/attack_runs.csv --output $(ARTICLE_OUTPUT)/summary/precision_assessment.csv
	$(PYTHON) scripts/verify_article_alignment.py --config $(ARTICLE_CONFIG) --results $(ARTICLE_OUTPUT)

article-report:
	$(PYTHON) -m tangle_sim report --input $(ARTICLE_OUTPUT) --language pt-BR

article-audit:
	$(PYTHON) -m tangle_sim audit --input $(ARTICLE_OUTPUT)

article-check:
	$(PYTHON) scripts/verify_article_alignment.py --config $(ARTICLE_CONFIG) --results $(ARTICLE_REFERENCE)

# Aliases mantidos para compatibilidade com instruções antigas.
paper: article

report: article-report
