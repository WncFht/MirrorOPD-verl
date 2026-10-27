import pytest
import torch

from verl.trainer.distillation.forward_vocab import forward_rows, full_forward_kl_result

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("vocab", [31, 4097, 151936])
def test_fused_rows_match_stable_reference(vocab):
    torch.manual_seed(42)
    teacher = torch.randn(3, vocab, device="cuda") * 2
    z = teacher + torch.randn_like(teacher) * 0.3
    extra = torch.randn(3, 128, device="cuda")
    ref = forward_rows(z.double(), teacher.double(), extra.double())
    actual = forward_rows(z, teacher, extra, fused=True)
    for x, y in zip(actual, ref, strict=True):
        torch.testing.assert_close(x.double(), y, rtol=4e-5, atol=2e-6)
    # Check quadratic values and non-vanishing gradients near equality.
    z = teacher + torch.randn_like(teacher) * 1e-4
    a = forward_rows(z, teacher, fused=True)
    r = forward_rows(z.double(), teacher.double())
    torch.testing.assert_close(a[0].double(), r[0], rtol=1e-3, atol=1e-11)
    torch.testing.assert_close(a[1].double(), r[1], rtol=1e-3, atol=1e-9)
    a = forward_rows(teacher, teacher, fused=True)
    assert torch.count_nonzero(a[0]) == torch.count_nonzero(a[1]) == 0


@pytest.mark.parametrize("backend", ["precompute", "precompute_fused", "recompute"])
def test_bfloat16_unequal_head_vjp(backend):
    torch.manual_seed(7)
    h = (torch.randn(9, 16, device="cuda") * 0.1).bfloat16().requires_grad_()
    w = (torch.randn(4097, 16, device="cuda") * 0.1).bfloat16().requires_grad_()
    t = (torch.randn(9, 24, device="cuda") * 0.1).bfloat16()
    v = (torch.randn(4225, 24, device="cuda") * 0.1).bfloat16()
    href = h.detach().double().requires_grad_()
    wref = w.detach().double().requires_grad_()
    lp = (href @ wref.T).log_softmax(-1)
    lq = (t.double() @ v[:4097].double().T).log_softmax(-1)
    ref = (lq.exp() * (lq - lp)).sum()
    ref.backward()
    result = full_forward_kl_result(h, w, t, v, backend=backend, token_chunk=4, teacher_support="student")
    result.loss_sum.backward()
    torch.testing.assert_close(result.loss_sum.double(), ref, rtol=1e-4, atol=1e-6)
    for got, expected in [(h.grad, href.grad), (w.grad, wref.grad)]:
        torch.testing.assert_close(got.double(), expected, rtol=0.04, atol=2e-6)


def test_fused_confident_rows_keep_tiny_loss_and_dominant_gradient():
    teacher = torch.tensor([[0.0, -20.0, -40.0], [0.0, -35.0, -45.0]], device="cuda")
    student = teacher.clone()
    student[:, 1] += 0.01
    ref = forward_rows(student.double(), teacher.double())
    fused = forward_rows(student, teacher, fused=True)
    torch.testing.assert_close(fused[0].double(), ref[0], rtol=1e-3, atol=1e-24)
    torch.testing.assert_close(fused[1].double(), ref[1], rtol=1e-3, atol=1e-22)
    assert (fused[0] > 0).all() and (fused[1][:, 0] != 0).all()


def test_fused_noncontiguous_rows_are_materialized_safely():
    z = torch.randn(2, 62, device="cuda")[:, ::2]
    teacher = torch.randn(2, 62, device="cuda")[:, ::2]
    actual = forward_rows(z, teacher, fused=True)
    expected = forward_rows(z.double(), teacher.double())
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.double(), b, rtol=4e-5, atol=2e-6)
    with pytest.raises(ValueError, match="equal shared-support"):
        forward_rows(z, teacher[:, :30], fused=True)
