from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class LocalLLM:
    def __init__(self, model_name_or_path: str, device: str = None, preload_model: bool = False):
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

        if preload_model:
            self._load_model()

    def _load_model(self):
        # Internal method to load the model and tokenizer.
        if not self.is_loaded:
            print(f"Loading model and tokenizer for '{self.model_name_or_path}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode for inference
            self.is_loaded = True
            print("Model and tokenizer loaded.")
        # else:
            # print("Model already loaded.")

    def _unload_model(self):
        # Internal method to unload the model and tokenizer and free up memory.
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
            outputs = self.model(**inputs)
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
            "top_k_predictions": top_k_results
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
        # According to Hugging Face Transformers' GenerationConfig, max_new_tokens must be > 0.
        # Handle this case early.
        if max_new_tokens <= 0:
            return ""

        # Ensure the model is loaded before attempting to generate text.
        if not self.is_loaded:
            print("Model not loaded. Loading now for this generation...")
            self._load_model()

        # Double-check that the tokenizer and model are available after attempting to load.
        if not self.tokenizer or not self.model:
            raise RuntimeError("Model or tokenizer not available. Call load() or use a 'with' statement.")

        # Tokenize the prompt text.
        # return_tensors="pt" ensures that the output are PyTorch tensors.
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        # Move tokenized inputs to the specified device (e.g., CUDA, MPS, CPU).
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Determine the End-Of-Sequence (EOS) token ID.
        # Some tokenizers might return a list of EOS token IDs (e.g., for multi-turn conversations).
        # We take the first one if it's a list, assuming it's the primary EOS token.
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        # Determine the padding token ID.
        # If pad_token_id is not explicitly set in the tokenizer,
        # it's often set to the eos_token_id for generation purposes.
        # This is crucial for some models when batching or using features like beam search,
        # as the model needs a defined pad_token_id for attention masking and generation.
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None and eos_token_id is not None:
            # print(f"Warning: pad_token_id not set for tokenizer {self.model_name_or_path}. Using eos_token_id ({eos_token_id}) as pad_token_id for this generation.")
            pad_token_id = eos_token_id


        # Prepare the GenerationConfig object.
        # This object centralizes all generation parameters, making it cleaner to manage.
        # It also allows for more advanced configurations and future compatibility.
        # The `max_new_tokens` parameter directly controls the maximum length of the generated sequence.
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,          # Maximum number of new tokens to generate after the prompt.
            do_sample=do_sample,                    # Whether to use sampling (True) or greedy decoding (False).
            temperature=temperature if do_sample else None, # Controls randomness in sampling. Only used if do_sample is True.
            top_k=top_k if do_sample else None,             # Filters to top-k most probable tokens. Only used if do_sample is True.
            top_p=top_p if do_sample else None,             # (Nucleus sampling) Filters to smallest set of tokens whose cumulative probability exceeds top_p. Only used if do_sample is True.
            num_beams=num_beams,                    # Number of beams for beam search. 1 means no beam search.
            repetition_penalty=repetition_penalty,  # Penalty for repeating tokens. 1.0 means no penalty.
            pad_token_id=pad_token_id,              # Sets the pad token ID for the generation.
            eos_token_id=eos_token_id,              # Sets the EOS token ID for the generation.
            **kwargs                                # Allows passing any other generation parameters supported by the model.
        )
        # Note: `GenerationConfig` intelligently handles None values for sampling-specific parameters
        # (temperature, top_k, top_p) when `do_sample` is False, so they don't interfere with greedy search.

        # Perform inference without calculating gradients to save memory and computation.
        with torch.no_grad():
            # Generate token sequences using the model's generate method.
            # `input_ids` and `attention_mask` are taken from the tokenized prompt.
            # `generation_config` bundles all other generation parameters.
            # The output_sequences will contain the token IDs of the prompt + generated text.
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],              # Input token IDs from the prompt.
                attention_mask=inputs["attention_mask"],    # Attention mask for the prompt.
                generation_config=generation_config         # The comprehensive generation configuration.
            )

        # Decode the generated tokens, ensuring to exclude the original prompt tokens.
        # We assume a batch size of 1 for this method's typical use case, hence `output_sequences[0]`.
        num_input_tokens = inputs["input_ids"].shape[1] # Get the number of tokens in the input prompt.
        # Slice `output_sequences` to get only the newly generated token IDs (after the prompt).
        generated_token_ids = output_sequences[0, num_input_tokens:]
        # Decode these token IDs back into a human-readable string.
        # `skip_special_tokens=True` removes tokens like [EOS], [PAD] from the final output.
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        return generated_text.strip() # Return the cleaned, generated text.

    def __enter__(self):
        """Context manager entry: loads the model."""
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: unloads the model."""
        self._unload_model()
