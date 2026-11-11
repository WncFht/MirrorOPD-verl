"""Coordinate gate for autoregressive trunk exports with optional MiMo MTP head."""


def compatible_trunk(base, target):
    if base.tied != target.tied:
        raise ValueError("Tied-weight convention differs")

    def trunk(checkpoint):
        return {
            k: v
            for k, v in checkpoint.shapes.items()
            if not k.startswith("model.mtp_layers.")
            and not (checkpoint.tied and k == "lm_head.weight" and "model.embed_tokens.weight" in checkpoint.shapes)
        }

    if trunk(base) != trunk(target):
        raise ValueError("Autoregressive trunk coordinates differ")
    extra_base = set(base.shapes) - set(target.shapes)
    extra_target = set(target.shapes) - set(base.shapes)
    for key in extra_base | extra_target:
        if not key.startswith("model.mtp_layers.") and key != "lm_head.weight":
            raise ValueError("Unexpected omitted tensor: " + key)
    return dict(
        omitted_mtp_base=sorted(k for k in extra_base if k.startswith("model.mtp_layers.")),
        omitted_mtp_target=sorted(k for k in extra_target if k.startswith("model.mtp_layers.")),
    )
