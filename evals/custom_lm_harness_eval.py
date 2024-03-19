from typing import List, Tuple
from tqdm import tqdm

import torch

from transformers import AutoTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.utils.eval import compute_loglikelihood_given_prompt_and_target

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("mamba")
class MambaEvalWrapper(LM):
    def __init__(
        self,
        pretrained: str = "state-spaces/mamba-370m",
        max_length=2048,
        batch_size=1,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        LM.__init__(self)

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

        self.model = MambaLMHeadModel.from_pretrained(
            pretrained, device=device, dtype=dtype
        )
        self.model.eval()
        self.model.max_length = int(max_length)
        self.model.batch_size = int(batch_size)

        self.max_length = int(max_length)
        self.device = torch.device(device)

    def loglikelihood(self, requests: List[Instance]):
        results = []
        with torch.no_grad():
            for instance in tqdm(requests):
                context, target = instance.arguments

                context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(
                    device=self.device
                )  # (1 x CONTEXT_LEN)
                if context == "":
                    context_ids = torch.Tensor([self.tokenizer.eos_token_id])
                assert (
                    len(context_ids.shape) == 2 and context_ids.shape[1] > 0
                ), "Expected at least one context token"

                target_ids = self.tokenizer(target, return_tensors="pt").input_ids.to(
                    device=self.device
                )  # (1 x TARGET_LEN)
                assert (
                    len(target_ids.shape) == 2 and target_ids.shape[1] > 0
                ), "Expected at least one target token"

                loglikelihood, is_greedy = (
                    compute_loglikelihood_given_prompt_and_target(
                        context_ids,
                        target_ids,
                        self.model,
                        self.max_length,
                        self.vocab_size,
                    )
                )
                results.append((loglikelihood, is_greedy))
        return results

    def generate_until(self, requests):
        raise NotImplementedError()

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
