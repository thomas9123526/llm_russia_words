# Chapter 37. Retrieval-Augmented Generation (RAG)

> **Part 8 — LLMs in the Real World** · Level: Advanced
>
> **In this chapter you will learn:** why a model's built-in knowledge is not enough · how a RAG pipeline indexes, retrieves and uses documents · where RAG fails and how to measure it

## What the model does not know

Ask a language model: "What does section 4.2 of our laboratory's internal safety rules say?" It has never seen that document, so it cannot know. If you are lucky, it says so. More often it produces a plausible but invented answer. The same happens with events after its **knowledge cutoff**, the date its training data ends, and with rare facts that appeared only a few times in the training corpus.

Everything a model learned during pre-training is stored in its weights. We call this **parametric knowledge**. It has three weaknesses: it is frozen at training time, it is hard to update (fine-tuning on new facts is expensive and unreliable, Chapter 31), and it gives no sources, so we cannot check where an answer came from.

**Retrieval-augmented generation (RAG)** adds a second, **non-parametric** memory: an external collection of documents that we can search at question time. The idea, named in the paper by Lewis et al. (2020), is simple: before the model answers, find the passages most relevant to the question and put them into the prompt. The model then answers by reading, rather than by remembering. It is the difference between a closed-book and an open-book exam.

This gives several advantages at once. The knowledge base can be updated by adding a document, with no training. Private data stays in your own index. And the answer can cite its sources, which makes it verifiable. RAG is one of the most widely used patterns for applying LLMs in practice.

## The pipeline: index, retrieve, generate

A RAG system has two stages. The first happens offline, once per document collection:

1. **Chunking.** Split each document into **chunks** of a few hundred tokens. Chunks must be small enough to be specific and to fit several into the context window, but large enough to keep meaning. They often overlap slightly, so that a sentence cut at a border is not lost.
2. **Embedding.** Turn every chunk into a vector with an embedding model (an encoder like those in Chapter 22, trained so that texts with similar meaning get nearby vectors).
3. **Indexing.** Store the vectors in an **index**, often called a **vector database**.

The second stage happens online, for every question:

4. **Retrieval.** Embed the question with the same model and find the $k$ chunks with the highest similarity.
5. **Generation.** Build a prompt containing the retrieved chunks and the question, and let the LLM answer.

The similarity is usually the **cosine similarity** we met in Chapter 5:

$$
\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{d} \rVert},
$$

where $\mathbf{q}$ is the question vector and $\mathbf{d}$ a chunk vector. If all vectors are normalized to length 1, the cosine is just the dot product, so retrieval over the whole collection becomes one matrix–vector multiplication.

The code below builds a complete toy pipeline. To keep it self-contained, the "embedding" is a normalized bag-of-words vector (Chapter 4) instead of a neural encoder, but every other step has the same shape as in a real system:

```python
import numpy as np

docs = [
    "Moscow is the capital of Russia.",
    "Seoul is the capital of South Korea.",
    "The transformer uses self-attention.",
    "LoRA adds low-rank matrices to frozen weights.",
]

def tokens(text):
    return text.lower().replace(".", "").replace("?", "").split()

vocab = sorted({w for d in docs for w in tokens(d)})
def embed(text):                        # toy embedding: bag-of-words vector
    v = np.array([tokens(text).count(w) for w in vocab], dtype=float)
    return v / (np.linalg.norm(v) + 1e-9)  # unit length, so dot = cosine

index = np.stack([embed(d) for d in docs])   # the "vector database"

def retrieve(question, k=2):
    scores = index @ embed(question)          # cosine similarity to every chunk
    top = np.argsort(-scores)[:k]
    return [(docs[i], round(float(scores[i]), 3)) for i in top]

question = "What is the capital of South Korea?"
hits = retrieve(question)
print(hits)
context = "\n".join(f"[{i+1}] {d}" for i, (d, _) in enumerate(hits))
prompt = f"Answer using only the sources.\n{context}\nQuestion: {question}\nAnswer:"
print(prompt)
```

The question has six words that appear in the vocabulary ("what" does not). The Seoul sentence shares all six, so its cosine is $6 / (\sqrt{6}\,\sqrt{7}) \approx 0.926$. The Moscow sentence shares four ("is the capital of"), giving $4 / (\sqrt{6}\,\sqrt{6}) \approx 0.667$. The final prompt numbers the sources, so the model can cite them as [1] and [2].

With a real neural encoder, "What is the main city of Korea?" would also find the Seoul chunk, though it shares few words with it. That is the strength of **dense retrieval**, introduced at scale by DPR (Karpukhin et al., 2020): two encoders map questions and passages into one vector space, trained so that a question lands near the passages that answer it.

## Making retrieval good

In a RAG system the generator can only be as good as what the retriever gives it. Several techniques improve retrieval.

**Sparse versus dense.** Classical **sparse retrieval** uses word overlap, weighted by rarity. The standard formula is **BM25**:

$$
\text{BM25}(D, Q) = \sum_{w \in Q} \text{IDF}(w) \cdot \frac{f(w, D)\,(k_1 + 1)}{f(w, D) + k_1\left(1 - b + b\,\frac{|D|}{\text{avgdl}}\right)},
$$

where $f(w, D)$ is how often word $w$ occurs in document $D$, $|D|$ is the document length, avgdl the average document length, IDF the inverse document frequency from Chapter 4, and $k_1$ (often about 1.2–2) and $b$ (often 0.75) are constants that control saturation and length normalization. BM25 is excellent for exact names, numbers and rare terms, where dense embeddings can be vague. Dense retrieval is better at paraphrases. **Hybrid search** combines both scores and often beats either alone.

**Approximate search.** With millions of chunks, comparing the question to every vector is too slow. **Approximate nearest neighbor search (ANN)** methods such as HNSW (a layered graph of neighbors) or the libraries built around inverted files and product quantization find almost the same top-$k$ much faster, at the cost of occasionally missing a true neighbor.

**Reranking.** The first-stage retriever is a **bi-encoder**: the question and the chunk are embedded separately, which is fast because chunk vectors are precomputed. But the two texts never "see" each other. A **cross-encoder** reads the question and a chunk together through one transformer and outputs a relevance score. It is far more accurate but too slow for the whole collection. So we retrieve, say, 50 candidates quickly and then **rerank** them with the cross-encoder, keeping the best 5.

**Query transformation.** A user's question is often short or ambiguous. The LLM itself can rewrite it, split a complex question into sub-questions, or generate a hypothetical answer and use its embedding for search (the HyDE method). This links naturally to the agents of Chapter 38, which decide for themselves when and what to search.

## Where RAG fails, and how to measure it

RAG reduces hallucination (Chapter 39) but does not remove it. Errors have two sources, and it is important to tell them apart:

- **Retrieval failures.** The right chunk is not in the top $k$: the question uses different words, the answer was split across two chunks, or the index is outdated. No generator can fix a missing fact.
- **Generation failures.** The right chunk is in the prompt, but the model ignores it, mixes it with its parametric knowledge, or misreads it. Liu et al. (2023) showed the **"lost in the middle"** effect: models use information at the beginning and end of a long context better than information in the middle. Order matters, so the best chunks are often placed first.

Two more problems are specific to RAG. When retrieved sources contradict each other or the model's own knowledge, the model must decide whom to trust, and its choice is not always the one we want. And retrieved text is untrusted input: a web page or document can contain instructions such as "ignore previous instructions and ...". This is **prompt injection** through retrieval, and an attacker who plants such text in the collection is performing **retrieval poisoning**. We return to this in Chapters 38 and 40.

To evaluate a RAG system, measure the two stages separately:

- For retrieval, **recall@k**: the fraction of questions for which at least one relevant chunk is among the top $k$. This requires a set of questions labeled with their relevant chunks.
- For generation, **faithfulness** (is every claim in the answer supported by the retrieved text?) and **answer correctness** (does the answer match the reference?). An answer can be faithful to wrong sources or correct by luck, without support; we want both.

In practice, faithfulness is often judged by another LLM that checks each claim against the sources, a method with its own biases that we discuss in Chapter 39. A good habit is to always keep a small, hand-labeled test set from your own domain, because general benchmarks say little about your documents.

RAG is not the only way to give a model new knowledge. Long-context models can sometimes read a whole document directly, and fine-tuning can teach style and domain format. In practice they combine well: fine-tuning teaches the model *how* to use sources, and retrieval supplies *which* sources.

## Summary

- Parametric knowledge in the weights is frozen, hard to update and gives no sources; RAG adds a searchable non-parametric memory.
- The pipeline: chunk documents, embed chunks, index them; at question time embed the question, retrieve the top-$k$ chunks and generate an answer from them.
- Dense retrieval handles paraphrases, sparse BM25 handles exact terms; hybrid search, approximate nearest neighbor indexes and cross-encoder reranking improve quality and speed.
- Failures come from retrieval (the right chunk is missing) or from generation (the model ignores or misreads it, e.g. "lost in the middle").
- Evaluate retrieval with recall@k and generation with faithfulness and correctness; treat retrieved text as untrusted input.

## Exercises

1. Explain in your own words the difference between parametric and non-parametric knowledge, and give one task where each is better.
2. In the code, compute by hand the cosine similarity between the question and "The transformer uses self-attention." Then run `retrieve(question, k=4)` and check your answer.
3. Run `retrieve("Which city is the Korean capital?")`. Which document comes first, and why does bag-of-words matching fail here (look at the words "Korean" and "Korea")? What would a dense encoder change?
4. You have a recall@5 of 0.95 but users still report wrong answers half of the time. Where would you look first, and which metric would you add?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| retrieval-augmented generation (RAG) | генерация, дополненная поиском | 검색 증강 생성 |
| knowledge cutoff | дата отсечения знаний | 지식 컷오프 |
| parametric knowledge | параметрические знания | 파라메트릭 지식 |
| chunk | фрагмент | 청크 |
| index | индекс | 인덱스 |
| vector database | векторная база данных | 벡터 데이터베이스 |
| cosine similarity | косинусное сходство | 코사인 유사도 |
| dense retrieval | плотный поиск | 밀집 검색 |
| hybrid search | гибридный поиск | 하이브리드 검색 |
| approximate nearest neighbor search | приближённый поиск ближайших соседей | 근사 최근접 이웃 탐색 |
| bi-encoder | би-энкодер | 바이 인코더 |
| cross-encoder | кросс-энкодер | 크로스 인코더 |
| reranking | переранжирование | 재순위화 |
| lost in the middle | эффект потери в середине контекста | 중간 정보 손실 |
| retrieval poisoning | отравление поисковой выдачи | 검색 오염 |

---
[← Previous: Efficient Inference: KV Cache, Quantization, Speculative Decoding](36-efficient-inference.md) · [Contents](README.md) · [Next: Agents and Tool Use →](38-agents-and-tools.md)
