# LLM-Studies
Useful resources which I'm studying related to LLMs and SLMs. 

---

## 1. Foundations of LLMs

### Natural Language Processing (NLP) Basics
- What is NLP?
- Common NLP tasks:
  - Named Entity Recognition (NER)
  - Sentiment Analysis
  - Text Classification

### Language Modeling
- What is a language model?
- Evolution of language models:
  - N-gram models
  - RNNs
  - LSTMs
  - Transformers

---

## 2. Core Concepts Behind LLMs

### Tokenization
- What are tokens?
- Tokenization methods:
  - Byte Pair Encoding (BPE)
  - WordPiece
  - SentencePiece
- Subword units and how tokenization impacts performance

### Attention Mechanism
- Why attention matters
- Self-attention vs. cross-attention
- “Attention Is All You Need” (Transformer introduction)

### Transformer Architecture
- Encoder vs. Decoder vs. Encoder-Decoder:
  - BERT (encoder)
  - GPT (decoder)
  - T5 (encoder-decoder)
- Architectural components:
  - Multi-head attention
  - Feed-forward networks
  - Residual connections
  - Layer normalization

### Training LLMs
- Pretraining:
  - Next-token prediction
  - Masked token prediction
- Finetuning:
  - Task-specific tuning
- RLHF (Reinforcement Learning with Human Feedback)

---

## 3. Embeddings and Retrieval

### Vector Embeddings
- What are embeddings?
- Evolution:
  - Word2Vec
  - GloVe
  - Transformer-based embeddings (e.g., BERT)
- Sentence/document embeddings via OpenAI, Cohere, HuggingFace models

### Vector Databases
- Approximate Nearest Neighbor (ANN) search
- Tools:
  - FAISS
  - Weaviate
  - Pinecone
  - Qdrant
  - Milvus
- Metadata filtering and hybrid dense+sparse search

---

## 4. Retrieval-Augmented Generation (RAG)

### What is RAG?
- Retriever + Generator architecture
- Comparison with:
  - Finetuning
  - Semantic search
  - Closed-book LLM QA

### Implementing RAG
- Document chunking (sentence-based, window-based, sliding windows)
- Embedding generation and storage
- Retrieval + prompt assembly

---

## 5. Efficient Training & Deployment

### LoRA (Low-Rank Adaptation)
- Parameter-efficient finetuning
- How LoRA works
- Tools:
  - HuggingFace PEFT
  - QLoRA

### Quantization
- Reducing model size (e.g., 4-bit, 8-bit)
- Trade-offs:
  - Speed
  - Accuracy
  - Memory usage

---

## 6. Reasoning and Tool-Use Frameworks

### Chain-of-Thought (CoT) Prompting
- Explicit reasoning steps
- Zero-shot vs. few-shot CoT

### ReAct (Reason + Act)
- Integrating reasoning and tool use
- Used in agent frameworks like LangChain, OpenAgents

### Toolformer / Function Calling
- Models that decide when/how to call tools
- Examples:
  - OpenAI function calling
  - Anthropic tool use
  - Toolformer paper

---

## 7. Frameworks and Tools for Development

### LangChain / LlamaIndex / Haystack
- Modular frameworks for RAG and agentic applications
- Chains, agents, memory, tool use

### OpenAI, HuggingFace, Cohere APIs
- Choosing the right provider
- Differences in capabilities and licensing

### Serving LLMs
- Using APIs vs. self-hosting
- Tools:
  - Ollama
  - vLLM
  - Text Generation Inference (TGI)
- GPU vs. CPU trade-offs
