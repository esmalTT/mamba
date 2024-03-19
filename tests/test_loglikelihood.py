
import pytest
import torch

from mamba_ssm.utils.eval import compute_loglikelihood, compute_loglikelihood_given_prompt_and_target 
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from transformers import AutoTokenizer

def test_loglikelihood():
    logits = torch.tensor([[0.5, 1.0],[1.5, 2.5]]).reshape(1, 2, 2)
    labels = torch.tensor([[0, 1]]).reshape(1, 2, 1)
    probs = torch.nn.functional.log_softmax(logits, dim=-1) # (B x L x VOCAB)
    assert compute_loglikelihood(logits, labels) == probs[0, 0, 0] + probs[0, 1, 1]

def test_loglikelihood_from_prompt():

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    weights = "state-spaces/mamba-370m"
    model = MambaLMHeadModel.from_pretrained(weights, device="cuda", dtype=torch.bfloat16)
    model.eval()

    max_length = 2048

    def compute(context, target):
        with torch.no_grad():
            context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device="cuda")  # (1 x CONTEXT_LEN)
            assert len(context_ids.shape) == 2 and context_ids.shape[1] > 0, "Expected at least one context token"

            target_ids = tokenizer(target, return_tensors="pt").input_ids.to(device="cuda")  # (1 x TARGET_LEN)
            assert len(target_ids.shape) == 2 and target_ids.shape[1] > 0, "Expected at least one target token"

            return compute_loglikelihood_given_prompt_and_target(context_ids, target_ids, model, max_length, tokenizer.vocab_size)

    llh1, greedy1  = compute("Mamba is the ", "x x x x x")
    llh2, greedy2  = compute("Mamba is the ", "something something something something something")
    llh3, greedy3  = compute("Mamba is the ", "this is really really wrong")
    llh4, greedy4  = compute("Mamba is the ", "first game to be released")
    llh5, greedy5  = compute("Mamba is the ", "first game to be released")

    assert llh1 < llh4, f"Expected {llh1} < {llh4}"
    assert llh2 < llh4, f"Expected {llh2} < {llh4}"
    assert llh3 < llh4, f"Expected {llh3} < {llh4}"
    assert llh4 == llh5, "Identical queries should match"
    assert greedy4 == greedy5

    print(llh1, llh2, llh3, llh4, greedy1, greedy2)

    a, _ = compute("Roof shingle removal: A man is sitting on a roof. He ", "is using wrap to wrap a pair of skis.")
    b, _ = compute("Roof shingle removal: A man is sitting on a roof. He ", "is ripping level tiles off.")
    c, _ = compute("Roof shingle removal: A man is sitting on a roof. He ", "is holding a rubik's cube.")
    d, _ = compute("Roof shingle removal: A man is sitting on a roof. He ", "starts pulling up roofing on a roof.")

    print(a, b, c, d)


if __name__ == "__main__":
    test_loglikelihood()
    test_loglikelihood_from_prompt()

