from __future__ import annotations

import torch
from torch import nn

from t5gemma2_vllm_plugin.vllm_adapter import T5Gemma2VllmForConditionalGeneration


class _ForwardRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_outputs = None

    def forward(self, input_ids, positions, inputs_embeds, encoder_outputs):
        self.encoder_outputs = encoder_outputs
        return torch.zeros_like(input_ids, dtype=torch.float32)


def _adapter() -> tuple[T5Gemma2VllmForConditionalGeneration, _ForwardRecorder]:
    adapter = T5Gemma2VllmForConditionalGeneration.__new__(
        T5Gemma2VllmForConditionalGeneration
    )
    nn.Module.__init__(adapter)
    model = _ForwardRecorder()
    adapter.model = model
    return adapter, model


def test_empty_encoder_outputs_list_normalizes_to_none() -> None:
    adapter, model = _adapter()

    adapter.forward(
        torch.ones((1, 2), dtype=torch.long),
        torch.arange(2),
        encoder_outputs=[],
    )

    assert model.encoder_outputs is None


def test_nonempty_encoder_outputs_list_is_concatenated() -> None:
    adapter, model = _adapter()
    first = torch.ones((1, 3))
    second = torch.full((2, 3), 2.0)

    adapter.forward(
        torch.ones((1, 2), dtype=torch.long),
        torch.arange(2),
        encoder_outputs=[first, second],
    )

    torch.testing.assert_close(model.encoder_outputs, torch.cat([first, second]))
