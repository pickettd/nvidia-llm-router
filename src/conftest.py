def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default=None,
        help="Base URL for the OpenAI-compatible endpoint (default: Ollama at localhost:11434)",
    )
    parser.addoption(
        "--router-url",
        default="http://0.0.0.0:8084/v1",
        help="Base URL for the Router Controller endpoint (default: http://0.0.0.0:8084/v1)",
    )
