from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalLLM:
    def __init__(self, model_name_or_path: str, device: str = None):
        self.model_name_or_path = model_name_or_path
        self.tokenizer = None
        self.model = None
        self.is_loaded = False

        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available(): # For Apple Silicon
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def _load_model(self):
        if not self.is_loaded:
            print(f"Loading model and tokenizer for '{self.model_name_or_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            self.is_loaded = True
            print("Model and tokenizer loaded.")
        # else:
            # print("Model already loaded.")

    def _unload_model(self):
        if self.is_loaded:
            print(f"Unloading model and tokenizer for '{self.model_name_or_path}'...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            print("Model and tokenizer unloaded.")
        # else:
            # print("Model not loaded, nothing to unload.")

    def load(self):
        """Explicitly loads the model."""
        self._load_model()

    def unload(self):
        """Explicitly unloads the model."""
        self._unload_model()

    def predict_next_token(self, text: str, top_k: int = 5) -> dict:
        if not self.is_loaded:
            print("Model not loaded. Loading now for this prediction...")
            self._load_model()
            # Decide if you want to unload immediately after if not managed by `with`
            # For now, let's assume if predict is called directly, it stays loaded
            # until an explicit `unload()` or program exit.
            # If this method is called inside a `with` block, `__exit__` will handle unloading.

        if not self.tokenizer or not self.model:
            raise RuntimeError("Model or tokenizer not available. Call load() or use a 'with' statement.")

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            all_logits = outputs.logits
            next_token_logits = all_logits[:, -1, :]
            next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

            # Top choice
            predicted_next_token_id = torch.argmax(next_token_probabilities, dim=-1)
            predicted_next_token_str = self.tokenizer.decode(predicted_next_token_id[0].item())
            predicted_next_token_prob = next_token_probabilities[0, predicted_next_token_id[0]].item()

            # Top K
            top_k_probs, top_k_indices = torch.topk(next_token_probabilities, top_k, dim=-1)

            top_k_results = []
            for i in range(top_k):
                token_id = top_k_indices[0, i].item()
                prob = top_k_probs[0, i].item()
                token_str = self.tokenizer.decode(token_id)
                top_k_results.append({"token_id": token_id, "token": token_str, "probability": prob})

        return {
            "input_text": text,
            "predicted_next_token": predicted_next_token_str,
            "predicted_next_token_probability": predicted_next_token_prob,
            "top_k_predictions": top_k_results,
            "attentions": [att.cpu() for att in outputs.attentions] if outputs.attentions is not None else None
        }

    def generate_text(self, prompt_text: str, max_new_tokens: int = 50, do_sample: bool = True, temperature: float = 1.0, top_k: int = 5, top_p: float = 0.95, num_beams: int = 1, repetition_penalty: float = 1.0, **kwargs) -> str:
        """
        Generates text based on a prompt.

        Args:
            prompt_text (str): The input text prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            do_sample (bool): Whether to use sampling; otherwise greedy decoding.
            temperature (float): Controls randomness in sampling. Lower is more deterministic.
            top_k (int): Filters to the top k most probable tokens for sampling.
            top_p (float): (Nucleus sampling) Filters to the smallest set of tokens whose cumulative probability exceeds top_p.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            repetition_penalty (float): Penalty for repeating tokens. 1.0 means no penalty.
            **kwargs: Additional arguments to pass to model.generate().

        Returns:
            str: The generated text (excluding the prompt).
        """
        if not self.is_loaded:
            print("Model not loaded. Loading now for this generation...")
            self._load_model()

        if not self.tokenizer or not self.model:
            raise RuntimeError("Model or tokenizer not available. Call load() or use a 'with' statement.")

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get eos_token_id from tokenizer, handle if it's a list
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        # Ensure pad_token_id is available for generation
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id


        # Prepare generation config
        # We can pass arguments directly, or use a GenerationConfig object
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None, # Temp only if sampling
            top_k=top_k if do_sample else None,             # Top-k only if sampling
            top_p=top_p if do_sample else None,             # Top-p only if sampling
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs # Pass through any other user-specified kwargs
        )
        # Remove None values from config to avoid issues if not sampling
        # This is mostly handled by GenerationConfig or model.generate() itself now
        # but good to be mindful.

        with torch.no_grad():
            # output_sequences will include the input prompt tokens
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config
            )

        # Decode the generated tokens, excluding the prompt
        # output_sequences[0] because we assume batch_size 1 for this method.
        num_input_tokens = inputs["input_ids"].shape[1]
        generated_token_ids = output_sequences[0, num_input_tokens:]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        return generated_text.strip()

    def __enter__(self):
        """Context manager entry: loads the model."""
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: unloads the model."""
        self._unload_model()
