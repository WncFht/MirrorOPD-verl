# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Matrix-free logit Jacobians and auditable Ritz geometry.

The operator is the derivative of the executed forward (including its precision
policy). Reverse-over-reverse AD differentiates only with respect to the dummy
logit cotangent, so it computes J v, not a parameter Hessian product.
"""

import torch

from .metrics import tensor_inner, tensor_norm_sq


class LogitJacobian:
    """Reusable pullback/pushforward for one retained forward graph."""

    def __init__(self, logits, parameters):
        self.logits = logits
        self.parameters = tuple(p for p in parameters if p.requires_grad)
        self.seed = None
        self.transpose_graph = None
        self.active = None

    def pullback(self, request):
        return logit_pullback(self.logits, self.parameters, request, retain_graph=True)

    def pushforward(self, direction):
        if len(direction) != len(self.parameters):
            raise ValueError("Parameter direction does not match Jacobian parameter order")
        # Build the differentiable transpose only for callers that need JVPs.
        if self.transpose_graph is None:
            self.seed = torch.zeros_like(self.logits, requires_grad=True)
            gradients = torch.autograd.grad(
                self.logits, self.parameters, self.seed, create_graph=True, retain_graph=True, allow_unused=True
            )
            self.active = tuple(i for i, g in enumerate(gradients) if g is not None)
            self.transpose_graph = tuple(gradients[i] for i in self.active)
        tangents = []
        for i, g in zip(self.active, self.transpose_graph, strict=True):
            value = direction[i]
            if value is None:
                value = torch.zeros_like(g)
            elif value.device != g.device or value.dtype != g.dtype:
                raise ValueError("Place parameter directions on the operator device/dtype before JVP")
            tangents.append(value)
        return torch.autograd.grad(self.transpose_graph, self.seed, tuple(tangents), retain_graph=True)[0].detach()

    def kernel(self, request):
        return self.pushforward(self.pullback(request))


def logit_pullback(logits, parameters, request, *, retain_graph=False):
    """Detached VJP; a source direction need not retain its forward graph."""
    parameters = tuple(p for p in parameters if p.requires_grad)
    gradients = torch.autograd.grad(logits, parameters, request.detach(), retain_graph=retain_graph, allow_unused=True)
    return tuple(None if g is None else g.detach() for g in gradients)


def direction_inner(left, right):
    """Reduce where gradients already reside; synchronize once for the result."""
    if len(left) != len(right):
        raise ValueError("Parameter directions differ in length")
    total = None
    for a, b in zip(left, right, strict=True):
        if a is None or b is None:
            continue
        value = tensor_norm_sq(a) if a is b else tensor_inner(a, b)
        total = value if total is None else total + value
    return float(total) if total is not None else 0.0


def response_summary(request, response):
    """Separate response magnitude from agreement with receiver's request."""
    b, y = request.double(), response.double()
    pressure, energy = float(b.square().sum()), float(y.square().sum())
    inner = float((b * y).sum())
    return dict(
        pressure=pressure,
        response_norm=energy**0.5,
        request_response_inner=inner,
        request_response_cosine=inner / (pressure * energy) ** 0.5 if pressure * energy > 0 else float("nan"),
        rayleigh=inner / pressure if pressure > 0 else float("nan"),
    )


def leading_spectrum(matvec, prototype, *, iterations=24, top=3, seed=20261109):
    """Fully reorthogonalized Krylov/Rayleigh-Ritz, with explicit residuals.

    Return only an estimated leading spectrum. The projected matrix is
    symmetrized; its measured asymmetry is saved, not silently discarded.
    """
    if iterations < top or top < 1:
        raise ValueError("Require iterations >= top >= 1")
    generator = torch.Generator(device=prototype.device).manual_seed(seed)
    q = torch.randn(prototype.shape, device=prototype.device, dtype=prototype.dtype, generator=generator)
    q /= q.norm()
    basis, images = [], []
    for _ in range(min(iterations, prototype.numel())):
        basis.append(q)
        image = matvec(q).detach()
        images.append(image)
        residual = image.double().clone()
        Q = torch.stack(basis, dim=1).double()
        for _ in range(2):
            residual -= Q @ (Q.T @ residual)
        norm = residual.norm()
        if norm <= 1e-10 * max(float(image.norm()), 1e-30):
            break
        q = (residual / norm).to(prototype.dtype)
    Q, Y = torch.stack(basis, dim=1).double(), torch.stack(images, dim=1).double()
    projection = Q.T @ Y
    asymmetry = float((projection - projection.T).norm() / projection.norm().clamp_min(1e-30))
    values, rotation = torch.linalg.eigh((projection + projection.T) / 2)
    order = torch.argsort(values, descending=True)
    values, rotation = values[order], rotation[:, order]
    count = min(top, len(values))
    vectors = Q @ rotation[:, :count]
    # Residuals use saved operator images, including any out-of-subspace motion.
    residuals = (Y @ rotation[:, :count] - vectors * values[:count]).norm(dim=0)
    return dict(
        eigenvalues=values.cpu(),
        vectors=vectors.to(prototype.dtype).cpu(),
        residual_relative=(residuals / values[:count].abs().clamp_min(1e-30)).cpu(),
        projected_asymmetry=asymmetry,
        iterations=len(basis),
        method="fully-reorthogonalized-krylov-rayleigh-ritz",
    )


def request_projection(request, spectrum):
    """Resolved leading request energy and its contribution to accessibility."""
    b = request.detach().double().cpu()
    pressure = float(b.square().sum())
    vectors = spectrum["vectors"].double()
    weights = (vectors.T @ b).square() / max(pressure, 1e-30)
    eigenvalues = spectrum["eigenvalues"][: len(weights)].double()
    return dict(
        leading_request_energy=weights.tolist(),
        leading_accessibility_contribution=(eigenvalues * weights).tolist(),
        unresolved_request_fraction=float(1 - weights.sum()),
    )
