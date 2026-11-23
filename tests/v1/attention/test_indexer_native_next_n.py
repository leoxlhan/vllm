# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which next_n the DSA indexer decode path may hand to DeepGEMM unflattened.

Getting this wrong is not a slow path but a crash: `fp8_fp4_paged_mqa_logits`
asserts both that the architecture implements the requested `next_n` and that
the schedule metadata was sized for the matching slot count.
"""

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _paged_mqa_logits_schedule_slots, has_deep_gemm
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.kv_cache_interface import MLAAttentionSpec

NUM_SMS = 114  # H100 PCIe


def _set_arch(monkeypatch, family: int, *, cuda: bool = True, deep_gemm: bool = True):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability // 10 == family,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: deep_gemm)


@pytest.mark.parametrize(
    "family,expected_native",
    [
        # SM90 gained next_n=4 (MTP=3) via 2-CTA multicast, but never 3.
        (9, {1, 2, 4}),
        # SM100 schedules any next_n with multi-atom tiles.
        (10, {1, 2, 3, 4, 5, 8}),
        # SM120 advertises multi-atom too but is unvalidated on hardware, so
        # it stays on the conservative gate. Loosen it only with measurements.
        (12, {1, 2}),
    ],
)
def test_native_decode_gate_per_architecture(monkeypatch, family, expected_native):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4, 5, 8):
        assert indexer._supports_native_decode(next_n) == (next_n in expected_native), (
            f"family={family} next_n={next_n}"
        )


@pytest.mark.parametrize(
    "cuda,deep_gemm", [(False, True), (True, False), (False, False)]
)
def test_native_decode_gate_without_deepgemm(monkeypatch, cuda, deep_gemm):
    """Without the DeepGEMM kernels only the shapes every backend handles."""
    _set_arch(monkeypatch, 9, cuda=cuda, deep_gemm=deep_gemm)
    assert [indexer._supports_native_decode(n) for n in (1, 2, 3, 4)] == [
        True,
        True,
        False,
        False,
    ]


def test_sm90_next_n_4_halves_the_schedule_slots(monkeypatch):
    """SM90 next_n=4 runs one scheduler task per 2-CTA cluster, not per SM."""
    _set_arch(monkeypatch, 9)
    assert _paged_mqa_logits_schedule_slots(NUM_SMS, 4) == NUM_SMS // 2
    for next_n in (1, 2, 3):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.parametrize("family", [10, 12])
def test_multicast_is_sm90_only(monkeypatch, family):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(90)
    or not has_deep_gemm(),
    reason="requires SM90 native DeepGEMM",
)
def test_native_mtp_schedule_on_padded_pcp_rank(dist_init):
    config = create_vllm_config(
        model_name="vllm-project/tiny-random-LlamaForCausalLM",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=128,
    )
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )
    # The uncompressed indexer builds its rank-local schedule without a
    # collective. dist_init supplies a real single-rank process group.
    config.parallel_config.prefill_context_parallel_size = 2
    device = torch.device("cuda", torch.accelerator.current_device_index())
    builder = indexer.DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        ),
        layer_names=["indexer"],
        vllm_config=config,
        device=device,
        block_table_width=2,
    )

    for width in (1, 2, 4):
        buf_addr = None
        for lengths in ([width, width], [0, 0], [width, 0], [width, width]):
            lens = torch.tensor(lengths, dtype=torch.int32)
            qsl = torch.cat([lens.new_zeros(1), lens.cumsum(0, dtype=torch.int32)])
            seq_lens = torch.where(lens > 0, 16 + lens, 0)
            common = CommonAttentionMetadata(
                query_start_loc=qsl.to(device),
                query_start_loc_cpu=qsl,
                seq_lens=seq_lens.to(device),
                num_reqs=2,
                num_actual_tokens=2 * width,
                max_query_len=max(lengths),
                max_seq_len=int(seq_lens.max()),
                block_table_tensor=torch.zeros(
                    (2, 2), dtype=torch.int32, device=device
                ),
                slot_mapping=torch.full(
                    (2 * width,), -1, dtype=torch.int64, device=device
                ),
                is_prefilling=torch.zeros(2, dtype=torch.bool),
            )
            decode = builder.build(0, common).decode
            torch.accelerator.synchronize()
            assert decode is not None
            expected = (seq_lens[:, None] - width + 1 + torch.arange(width)).clamp(
                min=0
            )
            torch.testing.assert_close(decode.seq_lens.cpu(), expected.to(torch.int32))
            torch.testing.assert_close(decode.decode_lens.cpu(), lens)
            assert decode.requires_padding == (0 < sum(lengths) < 2 * width)
            if buf_addr is None:
                buf_addr = decode.seq_lens.data_ptr()
            assert decode.seq_lens.data_ptr() == buf_addr
