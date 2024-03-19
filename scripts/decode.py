import argparse
import pickle

import torch
import torch.nn.functional as F

import random
import numpy as np

from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams

from typing import DefaultDict, List, Tuple
from collections import defaultdict
from functools import partial


def save_activations(
    activations: DefaultDict,
    name: str,
    module: torch.nn.Module,
    inp: Tuple,
    out: torch.Tensor,
) -> None:
    if torch.is_tensor(out):
        activations[name].append(out.detach().cpu())


def register_activation_hooks(
    model: torch.nn.Module,
) -> DefaultDict[str, List[torch.Tensor]]:
    activations = defaultdict(list)
    for name, module in model.named_modules():
        module.register_forward_hook(partial(save_activations, activations, name))
    return activations


def serialize_and_save_to_disk(filename: str, activation: DefaultDict[str, List[torch.Tensor]]) -> None:
    with open(filename, 'wb') as handle:
            pickle.dump(activation, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_through_decode(
    model,
    tokenizer,
    prompt: str,
    n_tokens_to_gen: int = 51,
    sample: bool = False,
    top_k: int = None,
):
    model.eval()
    inference_params = InferenceParams(max_seqlen=2046, max_batch_size=1)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device="cuda")
    prompt_token_counts = len(input_ids[0])
    promt_plus_generated_n_tokens = prompt_token_counts + n_tokens_to_gen - 1
    for token_n in range(promt_plus_generated_n_tokens):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(
                indices_to_input[:, token_n].unsqueeze(1),
                inference_params=inference_params,
            ).logits
            inference_params.seqlen_offset += 1

        probs = F.softmax(next_token_logits, dim=-1)
        (batch, _, vocab_size) = probs.shape

        if top_k is not None:
            (values, _) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=-1, keepdims=True)

        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]
        next_indices = next_indices.squeeze(1)
        if token_n >= prompt_token_counts - 1:
            input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]

    return output_completions


parser = argparse.ArgumentParser(description="Decode testing")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-370m")
parser.add_argument("--prompt", type=str, default="Mamba is the")
parser.add_argument("--genlen", type=int, default=10)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

np.random.seed(0)
random.seed(0)
torch.random.manual_seed(0)

device = "cuda"
dtype = torch.bfloat16

print(f"Loading model {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)

model.eval()

act = register_activation_hooks(model)
print(
    generate_through_decode(
        model, tokenizer, prompt=args.prompt, n_tokens_to_gen=args.genlen
    )
)
serialize_and_save_to_disk(f"{args.model_name.replace('/', '_')}_{args.genlen}_{dtype}_{args.prompt.replace(' ', '_')}.pickle", act)
