# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pytest_mock import MockerFixture
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import FinishReason
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.request import Request

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_preprocess_add_request_preserves_omni_fields():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
        payload_sender_info=None,
    )
    scheduler_request = SimpleNamespace()

    with patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    ):
        result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}


def test_abort_request_waits_for_transfer_drain_and_rejects_early_id_reuse(mocker: MockerFixture):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.output_queue = Queue()
    request = Request("internal", [1], SamplingParams(max_tokens=1), None, client_index=2)
    request.external_req_id = "external"
    adapter = mocker.Mock(spec=OmniChunkTransferAdapter)
    adapter.connector = mocker.Mock(supports_exact_key_cleanup=True)
    drained: Future[None] = Future()
    adapter.retire_transfer.return_value = drained
    adapter.is_transfer_retiring.return_value = True
    engine.scheduler = mocker.Mock(
        spec=OmniARScheduler, requests={request.request_id: request}, chunk_transfer_adapter=adapter
    )

    abort = mocker.patch.object(engine, "abort_requests")
    add = mocker.patch.object(EngineCoreProc, "add_request")
    acknowledgement = engine.abort_request_and_drain("external", "cancel", [])
    adapter.retire_transfer.assert_called_once_with("external", "cancel")
    abort.assert_called_once_with(["internal"])
    assert not acknowledgement.done()
    engine.add_request(request)
    add.assert_not_called()
    client_index, output = engine.output_queue.get_nowait()
    assert client_index == 2 and output.outputs[0].finish_reason == FinishReason.ABORT
    drained.set_result(None)
    assert acknowledgement.result(timeout=1) is True
    engine.reclaim_request_transfer("external", "cancel")
    engine.release_request_transfer("external", "cancel")
    adapter.reclaim_transfer.assert_called_once_with("external", "cancel")
    adapter.release_transfer.assert_called_once_with("external", "cancel")
    adapter.is_transfer_retiring.return_value = False
    engine.add_request(request)
    add.assert_called_once_with(request, 0)
