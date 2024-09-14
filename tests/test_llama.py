import pytest
import torch
from transformers import LlamaForCausalLM

from modelling import Llama


@pytest.mark.parametrize(
    "model_id",
    ["TinyLlama/TinyLlama_v1.1", "nvidia/Llama-3.1-Minitron-4B-Width-Base"],
)
def test_llama_correctness(model_id: str):
    inputs = torch.randint(0, 32_000, size=(1, 128))

    model = Llama.from_hf(model_id, dtype=torch.float32).eval()
    with torch.no_grad():
        actual = model(inputs)
    del model

    model_ref = LlamaForCausalLM.from_pretrained(model_id).eval()
    with torch.no_grad():
        expected = model_ref(inputs).logits
    del model_ref

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
