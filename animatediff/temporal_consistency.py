# temporal_consistency.py
import torch
import torch.nn.functional as F


def isinstance_str(x, cls_name):
    for _cls in type(x).__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def register_extended_attention(unet):
    """
    Replace spatial self-attention (attn1) on all BasicTransformerBlocks
    so every frame's Q attends to all frames' concatenated K/V.
    Cross-attention (attn2) is left completely unchanged.
    Skips AnimateDiff temporal blocks.
    """
    def sa_forward(self):
        to_out = self.to_out
        if isinstance(to_out, torch.nn.ModuleList):
            to_out = to_out[0]

        def forward(x, encoder_hidden_states=None, attention_mask=None, **kwargs):
            batch_size, sequence_length, dim = x.shape
            is_cross = encoder_hidden_states is not None
            enc = encoder_hidden_states if is_cross else x
            h   = self.heads
            hd  = dim // h

            q = self.to_q(x)
            k = self.to_k(enc)
            v = self.to_v(enc)

            if is_cross:
                # Cross-attention: standard, no changes
                q_ = q.view(batch_size, sequence_length, h, hd).transpose(1, 2)
                k_ = k.view(batch_size, -1, h, hd).transpose(1, 2)
                v_ = v.view(batch_size, -1, h, hd).transpose(1, 2)
                out = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=0.0)
                out = out.transpose(1, 2).reshape(batch_size, sequence_length, dim)
            else:
                # Self-attention: each frame attends to ALL frames' K/V.
                # After AnimateDiff's internal reshape, batch_size = B * F.
                # With separate CFG passes (B=1), batch_size = F directly.
                #
                # K_ext: (F, seq, dim) → flatten → (1, F*seq, dim)
                #                      → repeat  → (F, F*seq, dim)
                # Each frame gets the same global K/V pool to attend over.
                k_ext = k.reshape(1, batch_size * sequence_length, dim)\
                         .repeat(batch_size, 1, 1)
                v_ext = v.reshape(1, batch_size * sequence_length, dim)\
                         .repeat(batch_size, 1, 1)

                q_ = q.view(batch_size, sequence_length, h, hd).transpose(1, 2)
                k_ = k_ext.view(batch_size, -1, h, hd).transpose(1, 2)
                v_ = v_ext.view(batch_size, -1, h, hd).transpose(1, 2)
                out = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=0.0)
                out = out.transpose(1, 2).reshape(batch_size, sequence_length, dim)

            return to_out(out)
        return forward

    patched = 0
    for name, module in unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            if "motion" in name or "temporal" in name:
                continue  # leave AnimateDiff temporal blocks untouched
            module.attn1.forward = sa_forward(module.attn1)
            patched += 1

    print(f"[ExtendedAttention] Patched {patched} spatial self-attention layers.")
    return unet


def unregister_extended_attention(unet):
    """Remove patches — restores default forward by deleting instance override."""
    for _, module in unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            if "forward" in module.attn1.__dict__:
                del module.attn1.__dict__["forward"]
