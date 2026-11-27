# Chapter 23. GPT and Decoder-Only Models, In-Context Learning

> **Part 5 — Language Models in Practice** · Level: Intermediate
>
> **In this chapter you will learn:** how the GPT line of models developed from GPT to GPT-3 · the decoder-only architecture and how to count its parameters · what in-context learning is and why it changed how we use language models

## A model that was never taught to translate

Give a large decoder-only language model the following text and ask it to continue:

```text
Translate English to Russian.
sea otter => морская выдра
cheese => сыр
peppermint => мята перечная
book =>
```

A good model will continue with *"книга"*. This is surprising. The model was never trained on a translation dataset with input–output pairs. It was trained only to predict the next token of ordinary text (Chapter 21). Yet from three examples in its input it recognizes the task and performs it — without any update to its weights. This ability is called **in-context learning**, and it is the main reason why decoder-only models such as GPT became the foundation of modern LLMs.

In this chapter we first look at how this family developed, then at its architecture, and finally at in-context learning itself.

## From GPT to GPT-3

The story has three steps, each from OpenAI.

**GPT** (Radford et al., 2018) — "Generative Pre-Training". A Transformer decoder was pre-trained with causal language modeling on a corpus of books and then fine-tuned for each task, much as BERT would be a few months later. The key message was that **generative pre-training** produces representations that transfer well to many tasks.

**GPT-2** (Radford et al., 2019) was the same idea at a larger scale — up to 1.5 billion parameters, trained on a large collection of web pages. The paper had a provocative title: *"Language Models are Unsupervised Multitask Learners"*. The authors showed that without any fine-tuning the model could already do a little summarization, translation or question answering if the task was written as text, for example by adding *"TL;DR:"* after an article to ask for a summary. This is **zero-shot** use: the task is described, but no examples are given.

**GPT-3** (Brown et al., 2020) scaled up again, to 175 billion parameters. Its paper, *"Language Models are Few-Shot Learners"*, studied systematically what happens when a few examples are placed in the prompt. Performance improved steadily with model size, and for the largest model **few-shot prompting** sometimes approached fine-tuned systems. After GPT-3 the field started to treat a single large pre-trained model as a general-purpose tool, controlled through its input text.

## The decoder-only architecture

Architecturally, a GPT model is the decoder of Chapter 19 without cross-attention: a stack of Transformer blocks with a **causal mask**. GPT-2 introduced the arrangement that most later models kept:

1. Token embeddings plus learned position embeddings (Chapter 17).
2. $L$ blocks, each with **pre-normalization**: $x \leftarrow x + \text{Attn}(\text{LN}(x))$, then $x \leftarrow x + \text{FFN}(\text{LN}(x))$ (Chapter 18).
3. A final LayerNorm.
4. A linear output layer that maps each hidden vector to **logits** over the vocabulary. Its weight matrix is shared with the token embedding matrix — this is called **weight tying**.

It is useful to be able to count the parameters of such a model. Let $d$ be the hidden size, $L$ the number of layers and $V$ the vocabulary size. In each block, attention has four $d \times d$ matrices ($W_Q, W_K, W_V, W_O$), which gives $4d^2$. The feed-forward network has hidden size $4d$ and two matrices of size $d \times 4d$, which gives $8d^2$. Ignoring the small biases and LayerNorm parameters:

$$N \approx 12 L d^2 + V d$$

The first term is the Transformer blocks, the second is the embedding matrix. The code below makes the count exact for GPT-2.

```python
def gpt_params(vocab, ctx, d, n_layers):
    d_ff = 4 * d                                 # FFN hidden size in GPT-2
    emb = vocab * d + ctx * d                    # token + learned position embeddings
    attn = 4 * d * d + 4 * d                     # W_Q, W_K, W_V, W_O with biases
    ffn = 2 * d * d_ff + d_ff + d                # two linear layers with biases
    ln = 2 * 2 * d                               # two LayerNorms (scale + shift)
    final_ln = 2 * d
    # the output layer reuses the token embedding matrix (weight tying)
    return emb + n_layers * (attn + ffn + ln) + final_ln

print(f"GPT-2 small : {gpt_params(50257, 1024, 768, 12) / 1e6:.1f}M")
print(f"GPT-2 medium: {gpt_params(50257, 1024, 1024, 24) / 1e6:.1f}M")
print(f"12*d^2 per block, d=768: {12 * 768**2 / 1e6:.2f}M")
```

The output is 124.4M and 354.8M — the well-known sizes of the two smallest GPT-2 models. Notice that for GPT-2 small the embedding matrix ($50257 \times 768 \approx 38.6$M) is almost a third of the model. In very large models the $12Ld^2$ term dominates, which is why the rough formula is often written as $N \approx 12 L d^2$.

### Generation as a loop

A decoder-only model generates **autoregressively**: it reads the prompt, predicts a distribution for the next token, chooses one token, appends it to the input and repeats. The loop stops when the model produces an **end-of-sequence token** or reaches a length limit. All tokens — prompt and generated — must fit into the **context window**, the maximum sequence length the model was trained on (1024 tokens for GPT-2). How exactly to *choose* a token from the distribution is the topic of Chapter 24. How to avoid recomputing the whole prefix at every step, using the KV cache, is the topic of Chapter 36.

## In-context learning

Let us now describe in-context learning more precisely. Suppose we have $k$ demonstration pairs $(x_1, y_1), \dots, (x_k, y_k)$ and a new input $x$. We write them all into one text, the **prompt**, and let the model continue. The answer is produced from

$$P_\theta(y \mid x_1, y_1, \dots, x_k, y_k, x)$$

where $\theta$ are the fixed, pre-trained parameters. There is no gradient step and no change of $\theta$: the "learning" happens entirely inside one forward pass, through attention over the examples. Depending on $k$ we speak of:

- **zero-shot** ($k = 0$): only an instruction, e.g. *"Translate English to Russian: book =>"*;
- **one-shot** ($k = 1$): one example;
- **few-shot** ($k$ small, typically up to a few dozen, limited by the context window).

Several properties of in-context learning are important in practice.

- **It depends on scale.** In the GPT-3 experiments small models gained little from examples, while the largest models gained a lot. Abilities that appear only beyond a certain scale are often called **emergent abilities**, although whether they really appear suddenly or only look sudden because of how we measure them is still discussed (Chapter 27).
- **It is sensitive to the prompt.** The order of examples, their format and even the choice of label words can change accuracy noticeably. Prompt design is therefore partly an empirical craft.
- **The model uses the format more than one might expect.** Experiments have shown that randomly changing the labels in demonstrations sometimes hurts less than expected: the examples tell the model a lot about the *format* and the *set of possible answers*, not only about the exact mapping.

Why does in-context learning work at all? There is no complete answer, but there are useful partial explanations. One view is that the web contains many texts with repeated patterns — lists, tables, question–answer pages — so predicting the next token well *requires* recognizing a pattern from earlier in the context and continuing it. Interpretability research has found specific attention mechanisms, called **induction heads** (Olsson et al., 2022), which implement the rule "if *A* was followed by *B* earlier, and *A* appears again, predict *B*". This simple copying mechanism is a building block of more general in-context behaviour. Another view treats in-context learning as implicit inference: the examples help the model identify which "task" or "document type" generated the text, and it then continues accordingly.

In-context learning has clear limits. The examples occupy the context window, every request must include them again, and the model does not remember anything after the request is finished. When a task is used many times, fine-tuning (Chapters 31–33) is often cheaper and more reliable. In modern practice both approaches are combined: models are first instruction-tuned so that zero-shot instructions work well, and examples are added to the prompt when needed.

## Why decoder-only models won

By now most large language models are decoder-only. Several reasons work together:

- **One interface for all tasks.** Any task can be expressed as text in, text out: classification, translation, code, dialogue. There is no need for task-specific heads as in Chapter 22.
- **Efficient training signal.** Causal LM produces a loss at every position (Chapter 21).
- **Simple architecture that scales.** One stack of identical blocks, no encoder–decoder split, is easy to train in parallel on many devices (Chapter 28).
- **Efficient generation.** Since past tokens never attend to future ones, their keys and values can be cached and reused at every step (Chapter 36).

The price is that each token sees only its left context. In practice, at large scale, this limitation matters much less than the advantages. In Chapter 25 we will see how modern open models refine the GPT recipe without changing its basic idea.

## Summary

- GPT pre-trained a Transformer decoder with causal LM and then fine-tuned it; GPT-2 showed zero-shot task transfer; GPT-3 showed strong few-shot in-context learning at 175B parameters.
- A decoder-only model is a stack of pre-norm Transformer blocks with a causal mask, a final LayerNorm and an output layer, often tied to the embeddings.
- The parameter count is approximately $12 L d^2 + V d$; GPT-2 small has about 124M parameters.
- In-context learning means solving a task from instructions and examples in the prompt, $P_\theta(y \mid \text{examples}, x)$, without updating the weights.
- In-context learning grows with scale, is sensitive to prompt format, and is partly explained by mechanisms such as induction heads.

## Exercises

1. Explain in your own words the difference between fine-tuning and in-context learning. What changes in each case, and what stays fixed?
2. Use `gpt_params` to estimate the size of a model with $L = 48$, $d = 1600$, $V = 50257$ and context 1024. Compare your result with the approximate formula $12Ld^2$.
3. For GPT-2 small, what fraction of all parameters is in the token embedding matrix? What happens to this fraction if $d$ and $L$ grow but $V$ stays the same?
4. Write a zero-shot, a one-shot and a three-shot prompt for classifying the sentiment of Russian movie reviews. Which parts of the prompt tell the model the *format* of the answer?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| decoder-only model | модель только с декодировщиком | 디코더 전용 모델 |
| generative pre-training | генеративное предобучение | 생성적 사전 학습 |
| autoregressive generation | авторегрессионная генерация | 자기회귀 생성 |
| in-context learning | обучение в контексте | 문맥 내 학습 |
| prompt | запрос (промпт) | 프롬프트 |
| zero-shot learning | обучение без примеров | 제로샷 학습 |
| few-shot prompting | промптинг с несколькими примерами | 퓨샷 프롬프팅 |
| context window | контекстное окно | 컨텍스트 창 |
| logits | логиты | 로짓 |
| weight tying | связывание весов | 가중치 공유 |
| end-of-sequence token | токен конца последовательности | 시퀀스 종료 토큰 |
| emergent abilities | эмерджентные способности | 창발적 능력 |
| induction head | индукционная голова | 유도 헤드 |

---
[← Previous: BERT and Encoder Models](22-bert-and-encoders.md) · [Contents](README.md) · [Next: Generating Text: Greedy, Beam, Temperature, Top-k, Top-p →](24-text-generation.md)
