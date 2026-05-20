.PHONY: install run test lint

install:
	pip install -r requirements.txt

run:
	streamlit run app/ui/streamlit_app.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/
	mypy app/ --ignore-missing-imports
