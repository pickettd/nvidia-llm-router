def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default=None,
        help="Base URL for the OpenAI-compatible endpoint (default: Ollama at localhost:11434)",
    )
