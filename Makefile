# Makefile for Agent Environment

IMAGE_NAME = agent-environment
VERSION = 1.2.4
GHCR_REPO = ghcr.io/scaleapi/mcp-atlas

.PHONY: get_submodules build run run-docker shell test run-mcp-completion push

get_submodules:
	git submodule update --init

# load .env, then create mcp_server_config.json, then run the agent-environment service with the ENABLED_SERVERS from .env
# note some bugs with with servers that directly reference /data dir in mcp_server_template.json, so docker is preferred.
run: get_submodules
	cd services/agent-environment && \
	uv run dotenv -f ../../.env run -- bash -c " \
		./entrypoint.sh && \
		uv run python -m uvicorn agent_environment.main:app --host 0.0.0.0 --port 1984 \
	"

run-docker: # run docker container for mcp servers (agent-environment service)
	docker run --rm -p 1984:1984 --env-file .env $(IMAGE_NAME):latest

build: get_submodules # builds agent-environment
	bash services/agent-environment/dev_scripts/get_submodule_shas.sh > services/agent-environment/data/repos/git_submodule_info.csv
	cd services/agent-environment && docker buildx build --platform linux/amd64 -t $(IMAGE_NAME) .
	docker tag $(IMAGE_NAME):latest $(IMAGE_NAME):$(VERSION)

shell: # shell for agent-environment
	docker run -it --rm --env-file .env $(IMAGE_NAME):latest bash


# Makefile for MCP Eval

# Run the MCP completion server (port 3000, http post endpoint at /v2/mcp_eval/run_agent)
# Note: This runs agent completions (not evaluation/scoring). For scoring, see mcp_evals_scores.py
run-mcp-completion: 
	cd services/mcp_eval && uv run python -m mcp_completion.main

# Build and push multi-arch image to ghcr.io
# Requires Docker, and may not work with Rancher Desktop
# First do: docker login ghcr.io
push: get_submodules
	bash services/agent-environment/dev_scripts/get_submodule_shas.sh > services/agent-environment/data/repos/git_submodule_info.csv
	@echo "--- Building and pushing multi-arch $(GHCR_REPO):$(VERSION) and :latest ---"
	cd services/agent-environment && docker buildx build --platform linux/amd64,linux/arm64 \
		-t $(GHCR_REPO):$(VERSION) \
		-t $(GHCR_REPO):latest \
		--push .
	@echo "✓ Successfully pushed to $(GHCR_REPO):$(VERSION)"