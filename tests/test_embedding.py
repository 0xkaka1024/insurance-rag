from types import SimpleNamespace

from app.core.config import Settings
from app.core.embedding import EmbeddingClient


class FakeOpenAI:
    def __init__(self):
        self.calls: list[list[str]] = []
        outer = self

        class _Embeddings:
            def create(self, model: str, input: list[str]):
                outer.calls.append(input)
                # 故意乱序返回，验证按 index 重排
                data = [
                    SimpleNamespace(index=i, embedding=[float(hash(t) % 97), 0.5])
                    for i, t in enumerate(input)
                ]
                return SimpleNamespace(data=list(reversed(data)))

        self.embeddings = _Embeddings()


def _client(fake: FakeOpenAI) -> EmbeddingClient:
    return EmbeddingClient(settings=Settings(_env_file=None), client=fake)


def test_embed_preserves_order():
    fake = FakeOpenAI()
    vecs = _client(fake).embed(["甲", "乙", "丙"])
    assert len(vecs) == 3
    assert vecs[0][0] == float(hash("甲") % 97)
    assert vecs[2][0] == float(hash("丙") % 97)


def test_embed_batches_large_input():
    fake = FakeOpenAI()
    texts = [f"t{i}" for i in range(70)]
    vecs = _client(fake).embed(texts)
    assert len(vecs) == 70
    assert len(fake.calls) == 3  # 32 + 32 + 6
    assert [len(c) for c in fake.calls] == [32, 32, 6]
