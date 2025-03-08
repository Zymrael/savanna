import glob
import os

import torch

torch.set_printoptions(precision=5)
LOGITS_DIR = "/lustre/fs01/portfolios/dir/users/jeromek/savanna-test/inference/4layer/MP8CP2/checkpoints/global_step0/logits"
REF_LOGITS_DIR = os.path.join(LOGITS_DIR, "ref")
TEST_LOGITS_DIR = os.path.join(LOGITS_DIR, "test")

REF_LOGITS_PATHS = sorted(glob.glob(os.path.join(REF_LOGITS_DIR, "*.pt")))
TEST_LOGITS_PATHS = sorted(glob.glob(os.path.join(TEST_LOGITS_DIR, "*.pt")))

assert len(REF_LOGITS_PATHS) == len(TEST_LOGITS_PATHS)

#print(f"{REF_LOGITS_PATHS=}")
for ref_path, test_path in zip(REF_LOGITS_PATHS, TEST_LOGITS_PATHS):
    assert os.path.basename(ref_path) == os.path.basename(test_path)
    ref_logits = torch.load(ref_path)
    test_logits = torch.load(test_path)
    print(f"Testing {os.path.basename(ref_path)}")
    print(f" -> {ref_logits.shape=} {ref_logits.view(-1).cpu().tolist()[:5]}")
    print(f" -> {test_logits.shape=} {test_logits.view(-1).cpu().tolist()[:5]}")
    assert torch.allclose(ref_logits, test_logits)
    print(f" -> passed")


