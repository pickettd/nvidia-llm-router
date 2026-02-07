FROM nvcr.io/nvidia/tritonserver:25.01-py3

COPY src/router-server/requirements-ollama-test.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
