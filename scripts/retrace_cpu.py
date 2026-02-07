"""Re-trace the router models on CPU for CPU-only deployment.

The pre-trained .pt files were traced on GPU, baking CUDA ops into the
computation graph. map_location="cpu" only moves tensor data — the traced
ops still reference CUDA kernels. This script re-traces from scratch on CPU.

Usage (via Docker, which has all dependencies):
    docker run --rm \
      -v ./routers-cpu-override:/out \
      -v ./scripts/retrace_cpu.py:/app/retrace_cpu.py \
      router-server:blackwell \
      python /app/retrace_cpu.py
"""
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class TracedModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super(TracedModel, self).__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map
        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
        logits = [
            self.heads[k](mean_pooled_representation)
            for k in range(len(self.target_sizes))
        ]
        return logits


class WrapperModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return torch.cat(outputs, dim=1)


def trace_and_save(output_path):
    print(f"Loading model config and tokenizer...")
    config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/prompt-task-and-complexity-classifier"
    )

    print("Loading pretrained weights...")
    model = TracedModel(
        target_sizes=config.target_sizes,
        task_type_map=config.task_type_map,
        weights_map=config.weights_map,
        divisor_map=config.divisor_map,
    ).from_pretrained("nvidia/prompt-task-and-complexity-classifier")

    # Stay on CPU
    model.eval()
    wrapped_model = WrapperModel(model)
    wrapped_model.eval()

    print("Tokenizing sample input...")
    prompt = "Write a Python script that uses a for loop."
    encoded_texts = tokenizer(
        [prompt],
        return_tensors="pt",
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )

    print(f"Tracing model on CPU...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped_model,
            (encoded_texts["input_ids"], encoded_texts["attention_mask"]),
        )

    print(f"Saving to {output_path}...")
    traced.save(output_path)
    print("Done.")


if __name__ == "__main__":
    # Both routers use the same model architecture and weights
    trace_and_save("/out/task_router/model.pt")
    trace_and_save("/out/complexity_router/model.pt")
