PYTHON := python3
SRC    := src

.PHONY: help train test evaluate physics

help:
	@echo "Available targets:"
	@echo "  make train     - Run training"
	@echo "  make test      - Run test"
	@echo "  make physics   - Run physics tests"
	@echo "  make evaluate   - Compare trained models (set models in eval_config.yaml)"

train:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode train

test:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode test

evaluate:
	$(PYTHON) $(SRC)/deconfliction_factory.py --mode evaluate
	
physics:
	$(PYTHON) $(SRC)/test_physics.py
