from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from template import get_chat_template
import time
import logging

logger = logging.getLogger(__name__)

# TODO: Add tools support
#  https://docs.mistral.ai/capabilities/function_calling/
#  https://huggingface.co/docs/transformers/main/chat_templating#introduction


class LanguageModel:
    def __init__(self, model_path=None):
        logger.debug("CUDA Available:", torch.cuda.is_available())
        logger.debug("Create tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.debug("Tokenizer created")
        model_start = time.time()
        logger.debug("Starting model")
        # TODO: Fix Some weights of the model checkpoint (...) were not used when initializing MistralForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda")
        logger.debug(f"Model started in {time.time()-model_start}s")
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        conversation,
        token_generated_callback=None,
        generation_done_callback=None,
        max_length=1000,
        temperature=0.05,
    ):
        eos_token_id = self.model.config.eos_token_id

        generate_start = time.time()
        logger.debug("Infering response")
        tokens = self.tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=get_chat_template(),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to("cuda")
        attention_mask = tokens["attention_mask"].to("cuda")
        logger.debug(f"Request tokenized in {time.time()-generate_start}s")

        result = []
        for i in range(max_length):
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1,
                attention_mask=attention_mask,
                temperature=temperature,
                do_sample=True,
            )
            if i == 0:
                logger.debug(
                    "Time to first token: %s seconds", time.time() - generate_start
                )

            input_length = input_ids.size(1)

            new_tokens = outputs[:, input_length:]

            input_ids = torch.cat([input_ids, new_tokens], dim=-1)
            # Update attention mask to include the new token (mark it as "non-padded")
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], new_tokens.size(1)),
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )

            if new_tokens.item() == eos_token_id:
                break

            decoded_word, has_space = self.decode_single_word(outputs, first_word=i == 0)

            if has_space:
                result.append(" ")
            result.append(decoded_word)
            if token_generated_callback:
                token_generated_callback(f"{' ' if has_space else ''}{decoded_word}")
            logger.debug("%d: [%s]", i, f"{' ' if has_space else ''}{decoded_word}")

        logger.debug(f"Response inferred in {time.time()-generate_start}s")
        joined_result = "".join(result)
        if generation_done_callback:
            generation_done_callback(joined_result)
        return joined_result
    
    def decode_single_word(self, outputs, first_word = False):
        sentencepiece_subword_tokenization = "▁"

        decoded_word = self.tokenizer.convert_ids_to_tokens(
            outputs[0], skip_special_tokens=True
        )[-1]
        has_space = decoded_word[0] == sentencepiece_subword_tokenization
        if first_word and has_space:
            decoded_word = decoded_word[1:]
            has_space = False
        elif has_space:
            decoded_word = decoded_word[1:]
        
        return decoded_word, has_space
