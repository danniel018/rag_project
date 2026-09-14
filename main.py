from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding

from chunking import SemanticChunking

with open("ht400_maintenance_manual.md", "r") as f:
    text = f.read()

document = Document(text=text)

embed_local = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")

# swap in MarkdownChunking() (no args needed) to switch strategy
strategy = SemanticChunking(embed_local)

chunks = strategy.chunk(document)

print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.text}")
