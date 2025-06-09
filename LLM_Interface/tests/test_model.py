import unittest
from unittest.mock import patch, MagicMock
import torch
from LLM_Interface.model import LocalLLM

@patch('LLM_Interface.model.AutoModelForCausalLM.from_pretrained')
@patch('LLM_Interface.model.AutoTokenizer.from_pretrained')
class TestLocalLLM(unittest.TestCase):

    def setUp(self):
        # Configure the mock tokenizer instance (self.mock_tokenizer_instance)
        # This is the object that AutoTokenizer.from_pretrained will be mocked to return.
        self.mock_tokenizer_instance = MagicMock()
        self.mock_tokenizer_instance.eos_token_id = 50256
        self.mock_tokenizer_instance.pad_token_id = 50256
        self.mock_tokenizer_instance.vocab_size = 30522
        self.mock_tokenizer_instance.return_value = { # Mocking __call__
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        self.mock_tokenizer_instance.decode = MagicMock(
            side_effect=lambda ids, skip_special_tokens: " ".join(["gen_token"] * len(ids)).strip()
        )

        # Configure the mock model instance (self.mock_model_instance)
        # This is the object that AutoModelForCausalLM.from_pretrained will be mocked to return.
        self.mock_model_instance = MagicMock()
        def mock_generate_side_effect(input_ids, attention_mask, generation_config):
            num_generated_tokens = generation_config.max_new_tokens
            if num_generated_tokens == 0:
                return input_ids
            new_token_ids = torch.ones((input_ids.shape[0], num_generated_tokens), dtype=torch.long) * 1234
            return torch.cat([input_ids, new_token_ids], dim=1)
        self.mock_model_instance.generate = MagicMock(side_effect=mock_generate_side_effect)
        self.mock_model_instance.to = MagicMock(return_value=self.mock_model_instance)
        self.mock_model_instance.eval = MagicMock()

    def _configure_from_pretrained_mocks(self, mock_autotokenizer_fp, mock_automodel_fp):
        """Helper to link from_pretrained mocks to configured instances and reset them."""
        mock_autotokenizer_fp.reset_mock()
        mock_automodel_fp.reset_mock()
        self.mock_tokenizer_instance.reset_mock() # Reset the instance too
        self.mock_model_instance.reset_mock()     # Reset the instance too

        # Link Auto*.from_pretrained to return the pre-configured instances
        mock_autotokenizer_fp.return_value = self.mock_tokenizer_instance
        mock_automodel_fp.return_value = self.mock_model_instance

        # Re-configure instance mocks as reset_mock clears them
        # Tokenizer instance
        self.mock_tokenizer_instance.eos_token_id = 50256
        self.mock_tokenizer_instance.pad_token_id = 50256
        self.mock_tokenizer_instance.vocab_size = 30522
        self.mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        self.mock_tokenizer_instance.decode = MagicMock(
            side_effect=lambda ids, skip_special_tokens: " ".join(["gen_token"] * len(ids)).strip()
        )
        # Model instance
        def mock_generate_side_effect(input_ids, attention_mask, generation_config):
            num_generated_tokens = generation_config.max_new_tokens
            if num_generated_tokens == 0: return input_ids
            new_token_ids = torch.ones((input_ids.shape[0], num_generated_tokens), dtype=torch.long) * 1234
            return torch.cat([input_ids, new_token_ids], dim=1)
        self.mock_model_instance.generate = MagicMock(side_effect=mock_generate_side_effect)
        self.mock_model_instance.to = MagicMock(return_value=self.mock_model_instance)
        self.mock_model_instance.eval = MagicMock()


    def test_preload_model_true(self, mock_autotokenizer_fp, mock_automodel_fp):
        self._configure_from_pretrained_mocks(mock_autotokenizer_fp, mock_automodel_fp)

        LocalLLM("fake_model_path", preload_model=True)

        mock_autotokenizer_fp.assert_called_once_with("fake_model_path")
        mock_automodel_fp.assert_called_once_with("fake_model_path")
        self.mock_model_instance.to.assert_called_once()
        self.mock_model_instance.eval.assert_called_once()

    def test_preload_model_false_then_generate(self, mock_autotokenizer_fp, mock_automodel_fp):
        self._configure_from_pretrained_mocks(mock_autotokenizer_fp, mock_automodel_fp)

        llm = LocalLLM("fake_model_path", preload_model=False)
        # At this point, from_pretrained mocks should not have been called
        mock_autotokenizer_fp.assert_not_called()
        mock_automodel_fp.assert_not_called()

        llm.generate_text("prompt", max_new_tokens=5)
        mock_autotokenizer_fp.assert_called_once_with("fake_model_path")
        mock_automodel_fp.assert_called_once_with("fake_model_path")
        self.mock_model_instance.to.assert_called_once()
        self.mock_model_instance.eval.assert_called_once()

    def _run_generate_text_test(self, num_tokens_to_generate, mock_autotokenizer_fp, mock_automodel_fp):
        self._configure_from_pretrained_mocks(mock_autotokenizer_fp, mock_automodel_fp)

        llm = LocalLLM("fake_model_path", preload_model=True)
        # Assertions for preloading
        mock_autotokenizer_fp.assert_called_once_with("fake_model_path")
        mock_automodel_fp.assert_called_once_with("fake_model_path")
        self.mock_model_instance.to.assert_called_once()
        self.mock_model_instance.eval.assert_called_once()

        # Reset generate mock for the specific call in generate_text
        self.mock_model_instance.generate.reset_mock()

        generated_text = llm.generate_text("Test prompt", max_new_tokens=num_tokens_to_generate)

        if num_tokens_to_generate > 0:
            self.mock_model_instance.generate.assert_called_once()
            args, kwargs = self.mock_model_instance.generate.call_args
            generation_config_arg = kwargs['generation_config']
            self.assertEqual(generation_config_arg.max_new_tokens, num_tokens_to_generate)
            expected_text = " ".join(["gen_token"] * num_tokens_to_generate)
            self.assertEqual(generated_text, expected_text)
        else:
            self.mock_model_instance.generate.assert_not_called()
            self.assertEqual(generated_text, "")

    def test_generate_text_max_tokens_5(self, mock_autotokenizer_fp, mock_automodel_fp):
        self._run_generate_text_test(5, mock_autotokenizer_fp, mock_automodel_fp)

    def test_generate_text_max_tokens_10(self, mock_autotokenizer_fp, mock_automodel_fp):
        self._run_generate_text_test(10, mock_autotokenizer_fp, mock_automodel_fp)

    def test_generate_text_max_tokens_0(self, mock_autotokenizer_fp, mock_automodel_fp):
        self._run_generate_text_test(0, mock_autotokenizer_fp, mock_automodel_fp)

    def test_generate_text_with_eos_pad_handling(self, mock_autotokenizer_fp, mock_automodel_fp):
        self._configure_from_pretrained_mocks(mock_autotokenizer_fp, mock_automodel_fp)

        # Specific setup for this test: tokenizer.pad_token_id is None
        self.mock_tokenizer_instance.pad_token_id = None
        # Ensure from_pretrained mock returns this modified instance
        # This is already handled by _configure_from_pretrained_mocks assigning the instance

        # We are testing behavior when preload_model=False, so reset from_pretrained mocks
        # after _configure_from_pretrained_mocks (which assumes they might be called by preload=True)
        mock_autotokenizer_fp.reset_mock()
        mock_automodel_fp.reset_mock()
        # Also reset calls on model instance that might have been made if preload was True
        self.mock_model_instance.to.reset_mock()
        self.mock_model_instance.eval.reset_mock()


        llm = LocalLLM("fake_model_path", preload_model=False)
        llm.generate_text("Test prompt", max_new_tokens=3)

        mock_autotokenizer_fp.assert_called_once_with("fake_model_path")
        mock_automodel_fp.assert_called_once_with("fake_model_path")

        self.mock_model_instance.generate.assert_called_once()
        args, kwargs = self.mock_model_instance.generate.call_args
        generation_config_arg = kwargs['generation_config']

        self.assertEqual(generation_config_arg.pad_token_id, self.mock_tokenizer_instance.eos_token_id)

        # Restore pad_token_id on the shared instance for subsequent tests
        self.mock_tokenizer_instance.pad_token_id = 50256


if __name__ == '__main__':
    unittest.main()
