# Chapter 22. BERT and Encoder Models

> **Part 5 — Language Models in Practice** · Level: Intermediate
>
> **In this chapter you will learn:** how BERT's input and architecture are organized · how one pre-trained encoder is fine-tuned for classification, tagging and question answering · the main members of the BERT family and where encoders are still used

## Reading tasks

Consider three everyday problems:

1. Is the review *"The battery dies after two hours"* positive or negative?
2. In *"Ivan Petrov works at Yandex in Moscow"*, which words are names of people, organizations and places?
3. Given a paragraph about the Volga and the question *"Where does the Volga flow into?"*, which part of the paragraph is the answer?

None of these tasks requires *writing* new text. They require *reading* well: each word must be understood in the context of the whole sentence, both what comes before and what comes after. In Chapter 21 we saw that masked language modeling trains exactly this skill. In this chapter we study the model that made it famous — **BERT** (Bidirectional Encoder Representations from Transformers, Devlin et al., 2019) — and the family of **encoder-only models** that followed it.

## Architecture and input format

BERT is simply a stack of Transformer encoder blocks (Chapter 18) without a causal mask. The original paper described two sizes:

| Model | Layers | Hidden size $d$ | Heads | Parameters |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | ~110M |
| BERT-large | 24 | 1024 | 16 | ~340M |

The interesting part is the input. Text is split by a WordPiece **tokenizer** (Chapter 3) with a vocabulary of about 30 thousand subwords. Then two **special tokens** are added:

- `[CLS]` ("classification") at the very beginning. Its output vector is used as a summary of the whole input.
- `[SEP]` ("separator") at the end of each segment. Many tasks take *two* texts — a question and a paragraph, or two sentences — and `[SEP]` marks the border between them.

For a sentence pair, the input looks like this:

```text
[CLS] the cat sat [SEP] it was tired [SEP]
  A    A   A   A    A   B   B    B     B
```

The second line shows the **segment embedding**: a learned vector $s_A$ for every token of the first text and $s_B$ for every token of the second. The input vector for the token at position $i$ is the sum of three embeddings:

$$h_i^{(0)} = e_{\text{tok}}(x_i) + e_{\text{pos}}(i) + e_{\text{seg}}(i)$$

Here $e_{\text{tok}}$ is the token embedding (Chapter 5), $e_{\text{pos}}$ is a learned position embedding (Chapter 17), and $e_{\text{seg}}$ is $s_A$ or $s_B$. After $L$ encoder layers we obtain one contextual vector $h_i^{(L)}$ for every token. Unlike the static vectors of Word2Vec, these vectors depend on context: *"bank"* in *"river bank"* and in *"bank account"* receives different representations.

Sentences in a batch have different lengths, so shorter ones are filled with a `[PAD]` token. This **padding** must not influence the result, so a **padding mask** tells attention to ignore those positions.

## Pre-training: MLM and next sentence prediction

BERT was pre-trained with two objectives at the same time.

- **Masked language modeling**, exactly as in Chapter 21: 15% of the positions, with the 80/10/10 rule.
- **Next sentence prediction** (NSP). The model receives a pair of segments A and B. In 50% of the examples B really follows A in the corpus; in the other 50% B is a random sentence from another document. A small classifier on top of $h_{\text{[CLS]}}$ predicts which case it is. The idea was to teach the model relations between sentences, which are useful for question answering and inference tasks.

Later work questioned NSP. **RoBERTa** (Liu et al., 2019) removed it, trained on much more data for longer, used larger batches, and chose new masked positions every time a sentence was seen (*dynamic masking*). The result was clearly better than the original BERT. The lesson was important for the whole field: many "architectural" improvements turn out to be smaller than the effect of simply training well and for longer — a theme we will return to in Chapter 27 on scaling laws.

## Fine-tuning: one encoder, many heads

The power of BERT comes from **transfer learning**: we pre-train once on unlabeled text, and then fine-tune the whole network, plus a small new layer, on a labeled dataset for a specific task. This small task-specific layer is called a **head**. Fine-tuning usually needs only thousands of labeled examples and a few epochs, because the encoder already "knows" the language.

The three reading tasks from the beginning of the chapter correspond to three standard heads.

**Sequence classification** (sentiment, topic, spam). We take the output vector of `[CLS]` and apply a linear layer and softmax:

$$p(y \mid x) = \operatorname{softmax}\left(W h_{\text{[CLS]}}^{(L)} + b\right)$$

Here $W \in \mathbb{R}^{C \times d}$ and $b \in \mathbb{R}^{C}$ are new parameters and $C$ is the number of classes. During pre-training, `[CLS]` attended to all tokens in every layer, so after fine-tuning its vector learns to collect exactly the information the task needs.

**Token classification** (**named entity recognition**, part-of-speech tagging). The same linear layer is applied to *every* output vector $h_i^{(L)}$, giving one label per token, for example `B-PER`, `I-PER`, `B-ORG`, `O`. Because a word can be split into several subwords, usually only the first subword of each word receives a label.

**Extractive question answering.** The question is segment A and the paragraph is segment B. The answer is a span of the paragraph, so the model must predict where it starts and where it ends. Two learned vectors $w_{\text{start}}, w_{\text{end}} \in \mathbb{R}^d$ give scores for every token:

$$P_{\text{start}}(i) = \frac{\exp(w_{\text{start}} \cdot h_i)}{\sum_j \exp(w_{\text{start}} \cdot h_j)}, \qquad P_{\text{end}}(i) = \frac{\exp(w_{\text{end}} \cdot h_i)}{\sum_j \exp(w_{\text{end}} \cdot h_j)}$$

The predicted answer is the span $(i, k)$ with $i \le k$ that maximizes $P_{\text{start}}(i) \cdot P_{\text{end}}(k)$.

## Code: a tiny BERT-style classifier

The model below has all the elements of BERT in miniature: three embeddings, an encoder without causal mask, a padding mask, and a classification head on `[CLS]`. It is untrained, so its outputs are random, but the shapes and the data flow are the real ones.

```python
import torch
import torch.nn as nn

class TinyBERT(nn.Module):
    def __init__(self, vocab=1000, d=64, heads=4, layers=2, max_len=32, n_classes=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)        # token embeddings
        self.pos = nn.Embedding(max_len, d)      # learned positions (Chapter 17)
        self.seg = nn.Embedding(2, d)            # segment A / segment B
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, activation="gelu",
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)      # classification head on [CLS]

    def forward(self, ids, segs, pad_mask):
        pos = torch.arange(ids.size(1)).unsqueeze(0)
        h = self.tok(ids) + self.pos(pos) + self.seg(segs)
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # no causal mask!
        return self.head(h[:, 0])                # the vector at position 0 is [CLS]

PAD, CLS, SEP = 0, 1, 2
# "[CLS] a b [SEP] c d e [SEP] [PAD]"
ids  = torch.tensor([[CLS, 15, 27, SEP, 40, 41, 42, SEP, PAD]])
segs = torch.tensor([[0,   0,  0,  0,   1,  1,  1,  1,   0]])
pad_mask = ids == PAD                            # True marks padding to be ignored
model = TinyBERT()
print(model(ids, segs, pad_mask).shape)          # torch.Size([1, 2])
```

To fine-tune it, you would compute `F.cross_entropy` between these logits and the true labels and run the training loop from Chapter 10. For token classification, replace `h[:, 0]` with `h` to get one prediction per token.

## The BERT family and where encoders live today

After BERT, many improved encoders appeared. A few are worth knowing by name:

- **RoBERTa** — the same architecture, better training recipe (see above).
- **ALBERT** — shares parameters across layers and factorizes the embedding matrix, which makes the model much smaller in memory.
- **DistilBERT** — a smaller student model trained by **knowledge distillation** to imitate BERT's outputs; it is faster and keeps most of the quality.
- **ELECTRA** — trained with replaced token detection (Chapter 21), so the loss covers every token.
- **DeBERTa** — represents content and position separately inside attention (*disentangled attention*).
- **mBERT** and **XLM-R** — multilingual encoders trained on many languages at once, including Russian and Korean (Chapter 30).

Why are encoders still used when large decoder models can do almost everything? There are two main reasons: cost and the nature of the task. A model with 100 million parameters is cheap enough to classify millions of documents per day on modest hardware. Many industrial tasks — filtering, tagging, routing, moderation — are reading tasks where generation is not needed.

The most important modern use is **semantic search**. If we average the output vectors of an encoder (*mean pooling*) or take $h_{\text{[CLS]}}$, we get one vector per sentence. Fine-tuned with **contrastive learning**, as in Sentence-BERT (Reimers and Gurevych, 2019), such a model maps sentences with similar meaning to nearby vectors. Search then reduces to finding nearest neighbours by cosine similarity (Chapter 5). This is the retrieval part of retrieval-augmented generation in Chapter 37.

What encoders cannot do well is write. Since every token sees every other token, there is no natural left-to-right order in which to produce text, and MLM never trained the model to generate a whole sequence. For generation we need the decoder-only models of the next chapter.

## Summary

- BERT is a stack of Transformer encoder blocks with bidirectional attention; BERT-base has 12 layers and about 110M parameters.
- The input is `[CLS]`, segment A, `[SEP]`, segment B, `[SEP]`; each input vector is the sum of token, position and segment embeddings.
- BERT was pre-trained with MLM and next sentence prediction; RoBERTa showed that NSP is not needed and that better training matters a lot.
- Fine-tuning adds a small head: on `[CLS]` for classification, on every token for tagging, and start/end vectors for extractive question answering.
- Encoders remain the cheap, strong choice for classification and semantic search, but they are not designed to generate text.

## Exercises

1. Explain in your own words why the output vector of `[CLS]` can serve as a summary of the whole input, even though the `[CLS]` token itself carries no meaning.
2. Estimate the size of BERT-base's token embedding matrix for a vocabulary of 30,000 and $d = 768$. What fraction of the ~110M parameters is it?
3. Change `TinyBERT` into a token classifier with 5 labels and check that the output shape for the example input is `(1, 9, 5)`.
4. In extractive QA, suppose $P_{\text{start}} = (0.1, 0.6, 0.3)$ and $P_{\text{end}} = (0.2, 0.3, 0.5)$ over three tokens. Which span $(i, k)$ with $i \le k$ is chosen, and what is its score?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| encoder-only model | модель только с кодировщиком | 인코더 전용 모델 |
| special token | специальный токен | 특수 토큰 |
| [CLS] token | токен [CLS] | [CLS] 토큰 |
| segment embedding | эмбеддинг сегмента | 세그먼트 임베딩 |
| padding | дополнение | 패딩 |
| padding mask | маска паддинга | 패딩 마스크 |
| next sentence prediction | предсказание следующего предложения | 다음 문장 예측 |
| transfer learning | перенос обучения | 전이 학습 |
| fine-tuning | дообучение | 미세 조정 |
| classification head | классификационная голова | 분류 헤드 |
| named entity recognition | распознавание именованных сущностей | 개체명 인식 |
| question answering | вопросно-ответная система | 질의응답 |
| knowledge distillation | дистилляция знаний | 지식 증류 |
| contrastive learning | контрастивное обучение | 대조 학습 |

---
[← Previous: Pre-training Objectives: Causal and Masked Language Modeling](21-pretraining-objectives.md) · [Contents](README.md) · [Next: GPT and Decoder-Only Models, In-Context Learning →](23-gpt-and-decoders.md)
