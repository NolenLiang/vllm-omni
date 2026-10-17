# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks for scheduled function rows; no model or CUDA execution."""

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration as Thinker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def thinker_shell(*, forced=None):
    thinker = Thinker.__new__(Thinker)
    torch.nn.Module.__init__(thinker)
    thinker._use_function_head, thinker._dtype = True, torch.float32
    thinker.function_head = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        thinker.function_head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
    thinker._sessions = {"one": {}, "other": {}}
    if forced is not None:
        thinker._sessions["one"].update(forced_function_token=forced, forced_function_tokens=[forced])
    return thinker


@pytest.mark.parametrize("valid_rows", [1, 3])
@pytest.mark.parametrize("forced", [None, 99])
def test_function_projection_ignores_trailing_padding(valid_rows, forced):
    thinker = thinker_shell(forced=forced)
    hidden = torch.zeros(valid_rows + 1, 2)
    hidden[valid_rows - 1] = torch.tensor([1.0, 0.0])
    hidden[-1] = torch.tensor([0.0, 1.0])
    envelope = thinker.make_omni_output(
        hidden, model_intermediate_buffer=[{"request_id": "one"}], request_token_spans=[(0, valid_rows)]
    )
    # The same real linear head chooses 0 on the scheduled row and 1 on
    # padding. The projection must never turn the padding row into a call.
    expected = 0 if forced is None else forced
    assert int(envelope.multimodal_outputs["nvc_function_token"]) == expected
    output = thinker.postprocess(hidden[:valid_rows], request_id="one", multimodal_outputs=envelope.multimodal_outputs)
    assert int(output["nvc_prev_function_token"]) == expected


@pytest.mark.parametrize("spans", [None, [], [(0, 3)], [(1, 1)], [(True, 1)], ["invalid"]])
@pytest.mark.parametrize("forced", [None, 99])
def test_missing_or_invalid_span_uses_request_local_fallback(spans, forced):
    thinker = thinker_shell(forced=forced)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kwargs = {"model_intermediate_buffer": [{"request_id": "one"}]}
    if spans is not None:
        kwargs["request_token_spans"] = spans
    envelope = thinker.make_omni_output(hidden, **kwargs)
    assert "nvc_function_token" not in envelope.multimodal_outputs
    output = thinker.postprocess(hidden[:1], request_id="one", multimodal_outputs=envelope.multimodal_outputs)
    assert int(output["nvc_prev_function_token"]) == (0 if forced is None else forced)


def test_multi_request_projection_keeps_each_local_row():
    thinker = thinker_shell()
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]])
    envelope = thinker.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": "one"}, {"request_id": "other"}],
        request_token_spans=[(0, 1), (1, 2)],
    )
    tokens = envelope.multimodal_outputs["nvc_function_token"]
    assert [int(token) for token in tokens] == [0, 1]
    for index, request_id in enumerate(["one", "other"]):
        output = thinker.postprocess(
            hidden[index : index + 1], request_id=request_id, multimodal_outputs=envelope.multimodal_outputs
        )
        assert int(output["nvc_prev_function_token"]) == index


def test_intermediate_prefill_does_not_publish_or_consume_a_forced_token():
    thinker = thinker_shell(forced=99)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    info = {
        "request_id": "one",
        "_omni_is_prefill": True,
        "_omni_num_computed_tokens": 0,
        "_omni_prompt_len": 4,
    }
    envelope = thinker.make_omni_output(hidden, model_intermediate_buffer=[info], request_token_spans=[(0, 2)])
    assert "nvc_function_token" not in envelope.multimodal_outputs
    assert thinker._sessions["one"]["forced_function_token"] == 99

    final_info = dict(info, _omni_num_computed_tokens=2)
    final = thinker.make_omni_output(hidden, model_intermediate_buffer=[final_info], request_token_spans=[(0, 2)])
    assert int(final.multimodal_outputs["nvc_function_token"]) == 99
    assert thinker._sessions["one"]["forced_function_token"] == 99


def test_mixed_prefill_keeps_the_other_requests_token():
    thinker = thinker_shell()
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    envelope = thinker.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {
                "request_id": "one",
                "_omni_is_prefill": True,
                "_omni_num_computed_tokens": 0,
                "_omni_prompt_len": 8,
            },
            {"request_id": "other"},
        ],
        request_token_spans=[(0, 1), (1, 2)],
    )
    tokens = envelope.multimodal_outputs["nvc_function_token"]
    assert tokens[0].numel() == 0
    assert int(tokens[1]) == 1


def test_single_row_legacy_projection_is_unambiguous():
    thinker = thinker_shell()
    envelope = thinker.make_omni_output(torch.tensor([[1.0, 0.0]]), model_intermediate_buffer=[{"request_id": "one"}])
    assert int(envelope.multimodal_outputs["nvc_function_token"]) == 0
