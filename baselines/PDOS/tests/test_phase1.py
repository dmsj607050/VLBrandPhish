from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.se import SEBlock
from pdos.domain_verification import LegalDomainDB, registered_domain, verify_domain
from pdos.similarity import cosine_similarity


def test_se_shape():
    block = SEBlock(512, 16)
    x = torch.randn(2, 512, 20, 20)
    assert block(x).shape == x.shape


def test_cosine_identical_is_one():
    x = torch.randn(2048)
    assert abs(cosine_similarity(x, x) - 1.0) < 1e-5


def test_registered_domain_and_subdomain():
    assert registered_domain("www.alibaba.com") == "alibaba.com"
    db = LegalDomainDB(ROOT / "data" / "legal_domain_db" / "domains.json")
    assert verify_domain("https://www.alibaba.com/path", "alibaba", db).matched
    assert not verify_domain("https://alibaba.example.com", "alibaba", db).matched
