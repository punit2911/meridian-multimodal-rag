.PHONY: install run demo test eval docker
install: ; python -m pip install -r requirements-dev.txt
run: ; python -m meridian --load-demo
test: ; python -m pytest -q
eval: ; python scripts/run_eval.py
docker: ; docker compose up --build
