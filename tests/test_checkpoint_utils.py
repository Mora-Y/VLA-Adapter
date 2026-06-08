import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

MODULE_PATH = Path(__file__).parents[1] / "prismatic" / "vla" / "checkpoint_utils.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_utils", MODULE_PATH)
checkpoint_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_utils)

ACTION_QUERIES_KEY = checkpoint_utils.ACTION_QUERIES_KEY
action_queries_checkpoint_path = checkpoint_utils.action_queries_checkpoint_path
extract_action_queries_state_dict = checkpoint_utils.extract_action_queries_state_dict
load_action_queries_state_dict = checkpoint_utils.load_action_queries_state_dict
save_action_queries_checkpoint = checkpoint_utils.save_action_queries_checkpoint


class TinyVLA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_queries = torch.nn.Embedding(2, 3)


class PeftLikeWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = TinyVLA()


class CheckpointUtilsTest(unittest.TestCase):
    def test_extract_action_queries_from_wrapped_model(self):
        model = PeftLikeWrapper()
        with torch.no_grad():
            model.base_model.model.action_queries.weight.copy_(torch.arange(6).view(2, 3))

        state_dict = extract_action_queries_state_dict(model)

        self.assertEqual(list(state_dict), [ACTION_QUERIES_KEY])
        self.assertTrue(torch.equal(state_dict[ACTION_QUERIES_KEY], torch.arange(6).view(2, 3)))

    def test_save_and_load_action_queries_checkpoint(self):
        source = PeftLikeWrapper()
        target = TinyVLA()
        with torch.no_grad():
            source.base_model.model.action_queries.weight.copy_(torch.full((2, 3), 7.0))
            target.action_queries.weight.zero_()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            save_action_queries_checkpoint(source, tmp_path, "10_checkpoint.pt")
            checkpoint = torch.load(action_queries_checkpoint_path(tmp_path, "10_checkpoint.pt"), weights_only=True)
            load_action_queries_state_dict(target, checkpoint)

        self.assertTrue(torch.equal(target.action_queries.weight, torch.full((2, 3), 7.0)))

    def test_load_action_queries_rejects_shape_mismatch(self):
        target = TinyVLA()

        with self.assertRaisesRegex(ValueError, "Action query shape mismatch"):
            load_action_queries_state_dict(target, {ACTION_QUERIES_KEY: torch.zeros(4, 5)})


if __name__ == "__main__":
    unittest.main()
