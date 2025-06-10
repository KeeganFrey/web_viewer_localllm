# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from LLM_Interface.model import LocalLLM # Import LocalLLM

def test_predict_next_token_with_attentions():
    """Tests the predict_next_token method of LocalLLM for attention output."""
    print("Running test_predict_next_token_with_attentions...")
    # Instantiate LocalLLM with a valid model name
    # Ensure the model name is one that supports attention output if specific
    # For example, "Qwen/Qwen3-0.6B" or another similar model.
    llm_instance = LocalLLM(model_name_or_path="Qwen/Qwen3-0.6B")

    # Load the model
    llm_instance.load()

    # Call predict_next_token with a sample input string
    sample_text = "Hello world"
    predictions = llm_instance.predict_next_token(text=sample_text)

    # Assert that the key "attentions" exists in the returned dictionary
    assert "attentions" in predictions, "Key 'attentions' not found in predictions dictionary."

    # Assert that the value of "attentions" is a list (or None if model doesn't output attentions)
    if predictions["attentions"] is not None: # Models might not always return attentions
        assert isinstance(predictions["attentions"], list), \
            f"Value of 'attentions' is not a list, got {type(predictions['attentions'])}."

        # Assert that each element in the "attentions" list is a torch.Tensor
        if predictions["attentions"]: # If the list is not empty
            for i, attention_layer in enumerate(predictions["attentions"]):
                assert isinstance(attention_layer, torch.Tensor), \
                    f"Element {i} in 'attentions' list is not a torch.Tensor. Got {type(attention_layer)}."
                print(f"Attention layer {i} tensor device: {attention_layer.device}")
    else:
        print("No attentions returned by the model, which might be expected for some configurations/models.")


    print("test_predict_next_token_with_attentions assertions passed (or attentions were None as expected).")

    # Unload the model
    llm_instance.unload()


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

interface_instance = LocalLLM("Qwen/Qwen3-0.6B") # Changed variable name
interface_instance.load() # Changed variable name
predictions_output = interface_instance.predict_next_token("2 + 2 is ", 5) # Changed variable name
print(predictions_output) # Changed variable name
interface_instance.unload() # Changed variable name

# Call the new test function
test_predict_next_token_with_attentions()