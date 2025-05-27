# Load model directly
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

    def __enter__(self):
        """Context manager entry: loads the model."""
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: unloads the model."""
        self._unload_model()


def single_prediction():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    text = "The sum of 2+2 is: "
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        # logits has shape: [batch_size, sequence_length, vocab_size]
        all_logits = outputs.logits

        # We are interested in the logits for the *next* token prediction,
        # which corresponds to the logits at the *last token position* of the input sequence.
        # Shape: [batch_size, vocab_size]
        next_token_logits = all_logits[:, -1, :]

        # Convert these logits to probabilities
        # Shape: [batch_size, vocab_size]
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

        # 1. Get the top choice (predicted next token ID)
        # Shape: [batch_size]
        predicted_next_token_id = torch.argmax(next_token_probabilities, dim=-1)

        # Decode the predicted next token ID
        # Assuming batch_size is 1 for this example. If batch_size > 1, you'd loop or index.
        predicted_next_token_str = tokenizer.decode(predicted_next_token_id[0].item())

        print(f"Input: '{text}'")
        print(f"Predicted next token: '{predicted_next_token_str}'")
        print(f"Probability of predicted next token: {next_token_probabilities[0, predicted_next_token_id[0]].item():.4f}")


        # 2. Get the top K probabilities and their corresponding token IDs
        top_k = 5
        # top_k_probs will have shape [batch_size, k]
        # top_k_indices will have shape [batch_size, k] (these are token IDs)
        top_k_probs, top_k_indices = torch.topk(next_token_probabilities, top_k, dim=-1)

        print(f"\nTop {top_k} next token predictions and their probabilities:")
        # Assuming batch_size is 1 for this example
        for i in range(top_k):
            token_id = top_k_indices[0, i].item()
            prob = top_k_probs[0, i].item()
            token_str = tokenizer.decode(token_id)
            print(f"- '{token_str}' (ID: {token_id}): {prob:.4f}")

interface = LocalLLM("Qwen/Qwen3-0.6B")
interface.load()
l = interface.predict_next_token("2 + 2 is ",5)
print(l)
interface.unload()