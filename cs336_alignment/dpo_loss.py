import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

# Use EXACTLY this template (unchanged)
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""

def _get_unconditional_log_prob_sum(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Sum log p(token_t | token_<t) over the whole sequence (unconditional logprob of x⊕y).
    Returns shape: (batch,)
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (B, L, V)

    shift_logits = logits[:, :-1, :].contiguous()   # predict token 1..L-1
    shift_labels = input_ids[:, 1:].contiguous()    # actual tokens 1..L-1

    # token-level negative log-likelihood
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    token_log_probs = -loss.view_as(shift_labels)   # (B, L-1)

    return token_log_probs.sum(dim=-1)              # (B,)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    DPO loss using the 'trick':
      Δθ = log πθ(x⊕yw) - log πθ(x⊕yl)
      Δref = log πref(x⊕yw) - log πref(x⊕yl)
      loss = -logsigmoid(beta * (Δθ - Δref))

    Uses EXACTLY ALPACA_TEMPLATE for formatting and appends EOS after response.
    """

    # 1) Format full strings with EXACT template
    text_chosen = ALPACA_TEMPLATE.format(prompt=prompt, response=response_chosen)
    text_rejected = ALPACA_TEMPLATE.format(prompt=prompt, response=response_rejected)

    # 2) Tokenize (no need for prefix_len at all)
    chosen_ids = tokenizer(text_chosen, return_tensors="pt", add_special_tokens=True)["input_ids"]
    rejected_ids = tokenizer(text_rejected, return_tensors="pt", add_special_tokens=True)["input_ids"]

    # 3) Append EOS token AFTER the response
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")
    eos = torch.tensor([[tokenizer.eos_token_id]], dtype=chosen_ids.dtype)
    chosen_ids = torch.cat([chosen_ids, eos], dim=1)
    rejected_ids = torch.cat([rejected_ids, eos], dim=1)

    # 4) Unconditional log-prob sums under each model
    logp_theta_chosen = _get_unconditional_log_prob_sum(lm, chosen_ids)      # (1,)
    logp_theta_rejected = _get_unconditional_log_prob_sum(lm, rejected_ids)  # (1,)

    with torch.no_grad():
        logp_ref_chosen = _get_unconditional_log_prob_sum(lm_ref, chosen_ids)      # (1,)
        logp_ref_rejected = _get_unconditional_log_prob_sum(lm_ref, rejected_ids)  # (1,)

    # 5) DPO (prompt cancels inside each Δ)
    delta_theta = logp_theta_chosen - logp_theta_rejected
    # move ref deltas onto lm device for arithmetic
    device_theta = next(lm.parameters()).device
    delta_ref = (logp_ref_chosen - logp_ref_rejected).to(device_theta)

    logits = beta * (delta_theta - delta_ref)
    loss = -F.logsigmoid(logits)

    return loss.squeeze()  # scalar
