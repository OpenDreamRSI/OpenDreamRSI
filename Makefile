.PHONY: help install demo compare fidelity generalize benchmark evidence test lint clean

help:
	@echo "make install     editable install with dev + task extras"
	@echo "make demo        one offline Dream-RSI run on the synthetic task"
	@echo "make compare     Dream-RSI vs Recursive Fixed Exploration"
	@echo "make fidelity    is a frozen history an exact simulator?"
	@echo "make generalize  do dreaming's scores hold on held-out worlds?"
	@echo "make benchmark   fixed vs dream across seeds, paired"
	@echo "make evidence    all three of the above"
	@echo "make test        run the test suite"
	@echo "make clean       remove runs/ and caches"

install:
	python3 -m pip install -e ".[dev,tasks]"

demo:
	python3 -m dreamrsi.cli run --config configs/synthetic.yaml

compare:
	python3 -m dreamrsi.cli compare --config configs/synthetic.yaml

fidelity:
	python3 -m dreamrsi.cli fidelity --config configs/synthetic.yaml

generalize:
	python3 -m dreamrsi.cli generalize --config configs/synthetic.yaml --holdout-last 1

benchmark:
	python3 -m dreamrsi.cli benchmark --config configs/synthetic.yaml --seeds 0 1 2 3 4 5 6 7

evidence: fidelity generalize benchmark

test:
	python3 -m pytest -q

clean:
	rm -rf runs .pytest_cache **/__pycache__ */__pycache__ __pycache__
