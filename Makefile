serve:
	uvicorn app.main:app --reload --port 8084

test:
	pytest tests/ -v

demo:
	python notebooks/demo.py
