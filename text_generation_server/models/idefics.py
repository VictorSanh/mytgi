import torch
import torch.distributed

from typing import List, Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoProcessor,
)

from text_generation_server.models import IdeficsCausalLM
from text_generation_server.models.custom_modeling.idefics_modeling import (
    IdeficsForVisionText2Text,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


class IDEFICSSharded(IdeficsCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        self.device, self.dtype = device, dtype

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        config.quantize = quantize
        config.vision_config.quantize = quantize

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        # tokenizer.bos_token_id = config.decoder_start_token_id

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device=device,
            dtype=dtype,
            process_group=self.process_group,
        )

        model = IdeficsForVisionText2Text(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(IdeficsCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values: Optional = None,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        from loguru import logger
        logger.info(f"{input_ids.size()}")

        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            pixel_values=torch.randn(4, 1, 3, 224, 224, device=self.device, dtype=self.dtype),
            image_attention_mask=torch.zeros(4, 1024, 1, device=self.device),
            use_cache=True,
        )

        return (
            outputs.logits,
            outputs.past_key_values,
        )
