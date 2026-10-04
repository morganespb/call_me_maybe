export UV_PROJECT_ENVIRONMENT = /sgoinfre/goinfre/Perso/$(USER)/call_me_maybe

ARGS=
MYPY_FLAGS= --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

ESC = $(shell printf '\033')
BOLD = $(ESC)[1m
PURPLE = $(ESC)[35m
RESET = $(ESC)[0m
install:
	@echo "$(BOLD)$(PURPLE) ===> Installing dependencies$(RESET)"
	@uv sync
	@echo " $(BOLD)$(PURPLE)===> Install done !$(RESET)"

run:
	@echo "$(BOLD)$(PURPLE) ===> Running ... $(ARGS)$(RESET)"
	@uv run python -m src $(ARGS)

debug:
	@echo "$(BOLD)$(PURPLE) ===> Running in debug mode (pdb): n = next, c = continue, q = quit$(RESET)"
	@uv run python -m pdb -m src $(ARGS)

clean:
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@rm -rf .mypy_cache
	@echo "$(BOLD)$(PURPLE) Cleaned ! $(RESET)"

lint:
	@echo "$(BOLD)$(PURPLE) ===> running flake8 ... $(RESET)"; flake8 .; f=$$?; \
	echo "$(BOLD)$(PURPLE) ===> running mypy...$(RESET)"; mypy . $(MYPY_FLAGS); m=$$?; \
	if [ $$f -eq 0 ] && [ $$m -eq 0 ]; then echo " Lint OK ! $(RESET)"; fi

lint-strict:
	@echo "$(BOLD)$(PURPLE) ===> running flake8 ... $(RESET)"; flake8 .; f=$$?; \
	echo "$(BOLD)$(PURPLE) ===> running mypy...$(RESET)"; mypy . --strict ; m=$$?; \
	if [ $$f -eq 0 ] && [ $$m -eq 0 ]; then echo " Lint OK ! $(RESET)"; fi

test:
	@echo " $(BOLD)$(PURPLE) ===> Decoder tests$(RESET)"
	@python3 test_decoder.py
	@echo " $(BOLD)$(PURPLE) ===> Generator test (loads the model)$(RESET)"
	@uv run python test_generator.py

.PHONY: install run debug clean lint lint-strict test



