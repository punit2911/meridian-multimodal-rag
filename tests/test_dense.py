from qdrant_client import QdrantClient

from meridian.index.dense import DenseIndex
from meridian.schemas import Chunk, Document, Modality


def test_qdrant_dense_index_search_and_model_filter():
    chunks = [
        Chunk(id="hx200-1", doc_id="manual-200", modality=Modality.TEXT, text="HX-200 impeller torque specification", content="HX-200 impeller torque specification"),
        Chunk(id="hx500-1", doc_id="manual-500", modality=Modality.TEXT, text="HX-500 impeller torque specification", content="HX-500 impeller torque specification"),
        Chunk(id="generic-1", doc_id="generic", modality=Modality.TEXT, text="General service safety procedure", content="General service safety procedure"),
    ]
    docs = {
        "manual-200": Document(id="manual-200", title="HX-200", filename="hx200.pdf", models=["HX-200"]),
        "manual-500": Document(id="manual-500", title="HX-500", filename="hx500.pdf", models=["HX-500"]),
        "generic": Document(id="generic", title="Safety", filename="safety.pdf"),
    }
    client = QdrantClient(location=":memory:")
    index = DenseIndex(chunks, docs=docs, vector_store="qdrant", qdrant_client=client, collection_name="test_chunks")

    hits = index.search("HX-200 impeller torque", limit=3, models=["HX-200"])
    assert {i for i, _ in hits} == {0, 2}
    vectors = index.vectors([0, 2])
    assert set(vectors) == {0, 2}
    assert vectors[0].shape == vectors[2].shape
