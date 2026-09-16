.PHONY: help install demo compare test lint clean

help:
	@echo "make install   editable install with dev + task extras"
	@echo "make demo      one offline Dream-RSI run on the synthetic task"
	@echo "make compare   Dream-RSI vs Recursive Fixed Exploration"
	@echo "make test      run the test suite"
	@echo "make clean     remove runs/ and caches"

install:
	python3 -m pip install -e ".[dev,tasks]"

demo:
	python3 -m dreamrsi.cli run --config configs/synthetic.yaml

compare:
	python3 -m dreamrsi.cli compare --config configs/synthetic.yaml

test:
	python3 -m pytest -q

clean:
	rm -rf runs .pytest_cache **/__pycache__ */__pycache__ __pycache__
