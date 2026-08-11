"""OOM must degrade the batch size, never the reported score.

Batches are length-sorted, so the examples that exhaust memory are the LONGEST
ones. Recording them as empty completions -- the correct handling for an API
failure -- charges the harness's hardware limit to the model, and does it
selectively against hard examples. Measured before this was fixed: an arm
scoring 94% compliance on its early batches reported 55% overall because ~40%
of examples, all long, generated nothing.
"""

from __future__ import annotations

from eval.run_eval import _generate_chunk, _is_oom


class _OomError(RuntimeError):
    pass


class FakeBackend:
    """Fails on any batch larger than `limit`, like a card that runs out of VRAM."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.batch_calls: list[int] = []

    def generate_batch(self, texts: list[str], few_shot) -> list[str]:
        self.batch_calls.append(len(texts))
        if len(texts) > self.limit:
            raise _OomError("CUDA error: out of memory")
        return [f"out:{t}" for t in texts]

    def generate(self, text: str, few_shot) -> str:
        self.batch_calls.append(1)
        if self.limit < 1:
            raise _OomError("CUDA error: out of memory")
        return f"out:{text}"


def _chunk(n: int) -> list[dict]:
    return [{"posting_id": f"p{i}", "source_text": f"t{i}"} for i in range(n)]


class TestOomClassification:
    def test_recognises_cuda_oom(self):
        assert _is_oom(RuntimeError("CUDA error: out of memory"))
        assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate ..."))

    def test_other_errors_are_not_oom(self):
        assert not _is_oom(ValueError("bad json"))
        assert not _is_oom(TimeoutError("read timed out"))


class TestBackoff:
    def test_halves_until_it_fits_and_loses_nothing(self):
        be = FakeBackend(limit=2)
        cap = [8]
        out = _generate_chunk(be, _chunk(8), [], cap)
        assert out == [f"out:t{i}" for i in range(8)], "every example must be generated"
        assert cap[0] <= 2, "cap must shrink to a size that fits"

    def test_cap_persists_so_later_batches_do_not_re_oom(self):
        """Length-sorted order means later batches are longer, never shorter."""
        be = FakeBackend(limit=2)
        cap = [8]
        _generate_chunk(be, _chunk(8), [], cap)
        shrunk = cap[0]
        be.batch_calls.clear()
        _generate_chunk(be, _chunk(8), [], cap)
        assert max(be.batch_calls) <= shrunk, "second chunk must not retry the failed size"

    def test_no_oom_means_no_shrink(self):
        be = FakeBackend(limit=99)
        cap = [8]
        out = _generate_chunk(be, _chunk(8), [], cap)
        assert len(out) == 8 and cap[0] == 8

    def test_non_oom_error_still_scores_empty(self):
        """A model whose backend errors is not thereby more accurate."""
        class Broken:
            def generate_batch(self, texts, few_shot):
                raise ValueError("malformed request")

        out = _generate_chunk(Broken(), _chunk(4), [], [4])
        assert out == ["", "", "", ""]

    def test_single_example_that_cannot_fit_is_reported_not_hidden(self):
        be = FakeBackend(limit=0)
        out = _generate_chunk(be, _chunk(1), [], [1])
        assert out == [""], "a genuine hardware limit is reportable, not silently dropped"
