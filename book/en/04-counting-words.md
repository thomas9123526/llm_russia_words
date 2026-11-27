# Chapter 4. Counting Words: Bag of Words, N-grams and Perplexity

> **Part 1 — Text and Representation** · Level: Beginner
>
> **In this chapter you will learn:** how to represent a document by word counts · how n-gram language models estimate probabilities by counting, and why they need smoothing · how perplexity measures the quality of a language model

## What can we learn just by counting?

In Chapter 1 we built a tiny model from three sentences and saw that it gives the sentence *"the dog saw the cat"* probability zero, because the pair *dog saw* never occurred. Yet the sentence is perfectly reasonable. In this chapter we take counting seriously. Counting words was the main method of language processing for decades, and its ideas — and its problems — explain why later methods look the way they do. We study two uses of counts: describing *what a document is about* (bag of words) and predicting *what comes next* (n-gram models). Then we learn how to measure how good a language model is.

## Bag of words

Imagine you must sort news articles into "sport" and "economy". A simple observation helps: sport articles contain words like *match*, *goal* and *team*; economy articles contain *bank*, *inflation* and *market*. The order of the words matters less than which words occur and how often.

The **bag-of-words** model represents a document by the counts of the words in it, ignoring their order — as if we put all the words into a bag and shook it. With a vocabulary of $V$ words, each document becomes a vector of length $V$, where position $i$ holds the number of times word $i$ occurs. For example, with the vocabulary *(bank, goal, match, rate, team)*, the text *"the team scored a goal, a second goal won the match"* becomes $(0, 2, 1, 0, 1)$. Most entries in such vectors are zero, because a document uses only a small part of the vocabulary; vectors like this are called **sparse vectors**.

Raw counts have a weakness: words like *the*, *и*, *в* are frequent in every document and tell us nothing about the topic. A classic correction is **TF-IDF** (term frequency – inverse document frequency):

$$\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \log \frac{D}{\text{df}(t)},$$

where $\text{tf}(t, d)$ is the number of times term $t$ occurs in document $d$, $D$ is the total number of documents, and $\text{df}(t)$ is the number of documents that contain $t$. A word that appears in every document has $\text{df}(t) = D$, so its weight is $\log 1 = 0$. A word that appears in few documents gets a large weight. TF-IDF vectors were (and still are) used in search engines and text classification.

The bag of words throws away word order completely: *"собака укусила человека"* and *"человек укусил собаку"* have almost the same counts but very different meanings. For a language model, which must predict the next word, order is essential. We need a different use of counts.

## N-gram language models

Recall from Chapter 1 that a language model must estimate $P(w_t \mid w_{<t})$, the probability of the next token given the *whole* context. We cannot count whole contexts: almost every long sentence is unique, so its count is 0 or 1. An **n-gram model** makes a simplifying assumption: the next word depends only on the last $n - 1$ words.

$$P(w_t \mid w_1, \dots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \dots, w_{t-1}).$$

This is called the **Markov assumption**, after the Russian mathematician Andrey Markov, who studied such chains of dependent events (and, famously, applied them to the letters of Pushkin's *Eugene Onegin* in 1913). An **n-gram** is a sequence of $n$ consecutive tokens. With $n = 1$ we have a **unigram** model (no context at all), with $n = 2$ a **bigram** model, and with $n = 3$ a **trigram** model.

How do we estimate the probabilities? The natural answer is to use relative frequencies. For a bigram model,

$$P(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})},$$

where $C(w_{t-1}, w_t)$ is the number of times the pair occurs in the corpus and $C(w_{t-1})$ is the number of times the first word occurs (as a previous word). This is exactly what we did in Chapter 1: $P(\text{cat} \mid \text{the}) = 2/6$. This choice is not arbitrary: it is the **maximum likelihood estimate** (MLE), the set of probabilities under which the training corpus is as probable as possible. We will meet maximum likelihood again in Chapter 8, where it becomes the loss function of neural networks.

The choice of $n$ is a trade-off. A larger $n$ uses more context, so predictions can be sharper: after *"столица Российской"* the next word is almost surely *"Федерации"*. But the number of possible n-grams grows very fast — with a vocabulary of $V$ words there are $V^n$ possible n-grams — and most of them never occur in any corpus. This is the problem of **data sparsity**. In practice, n-gram models rarely used $n$ above 5.

## Zero counts and smoothing

Data sparsity causes a serious failure. If an n-gram never occurred in training, its MLE probability is zero. By the chain rule, a single zero factor makes the probability of the whole sentence zero. Our model then claims that *"the dog saw the cat"* is impossible, only because it never saw *dog saw*.

The fix is **smoothing**: take a little probability from the n-grams we have seen and give it to those we have not. The simplest method is **Laplace smoothing**, or add-one smoothing: pretend that every possible pair occurred one more time than it did.

$$P_{\text{add-}k}(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t) + k}{C(w_{t-1}) + k V}.$$

Here $k = 1$ gives Laplace smoothing, and $V$ is the number of possible next words. We add $kV$ to the denominator because we added $k$ to each of the $V$ possible numerators; this keeps the probabilities summing to 1. Smaller values of $k$ (such as 0.1) are often better, because adding a whole count to every unseen pair takes too much probability from the seen ones.

Better smoothing methods were developed for practical systems. **Backoff** uses a shorter context when the longer one has no data: if the trigram is unknown, use the bigram. **Interpolation** mixes trigram, bigram and unigram estimates with weights. **Kneser–Ney smoothing**, the best-known classical method, also takes into account in how many *different* contexts a word appears. We do not need their details, but the main idea is worth remembering: a model must spread probability over events it has never seen.

## Perplexity: how good is a language model?

Suppose we have two language models. Which one is better? We need a number. The standard idea is to take a **test set** — text that the model did *not* see during training — and ask how probable the model finds it. A better model assigns higher probability to real text.

The raw probability of a test text is inconvenient: it is a product of many numbers smaller than 1, so it becomes tiny and depends on the length of the text. We therefore take the logarithm, average it per token, and turn it back into an understandable scale. The result is called **perplexity**:

$$\text{PPL} = \exp\left( -\frac{1}{N} \sum_{t=1}^{N} \log P(w_t \mid w_{<t}) \right).$$

Here $N$ is the number of predicted tokens in the test text, $\log$ is the natural logarithm, and $\exp$ is its inverse. The expression inside the brackets, without the exponent, is the average negative log-probability per token; it is the **cross-entropy** between the text and the model, and in Chapter 8 it will become our training loss. Lower perplexity means a better model.

Perplexity has a nice interpretation. Suppose a model is completely uninformed and gives every one of $V$ words the same probability $1/V$. Then every term in the sum is $\log(1/V)$, and $\text{PPL} = \exp(\log V) = V$. So perplexity is the *effective number of choices* the model hesitates between at each step. A perplexity of 20 means that the model is, on average, as uncertain as if it were choosing uniformly among 20 words.

Let us compute it. The code below trains a bigram model with add-k smoothing on our three sentences and evaluates it on *"the dog saw the cat"*.

```python
import math
from collections import Counter

train = ["the cat sat on the mat", "the dog sat on the rug", "the cat saw the dog"]
test = "the dog saw the cat"

def tokens(s):
    return ["<s>"] + s.split() + ["</s>"]

history, bigrams = Counter(), Counter()
for s in train:
    t = tokens(s)
    history.update(t[:-1])                 # how often each word is a "previous" word
    bigrams.update(zip(t, t[1:]))          # how often each pair occurs
V = len({w for s in train for w in tokens(s)}) - 1   # possible next words (no <s>)

def prob(prev, word, k=1.0):
    # add-k smoothing; k = 1 is Laplace smoothing, k = 0 is plain counting
    return (bigrams[(prev, word)] + k) / (history[prev] + k * V)

print("unsmoothed P(saw | dog) =", prob("dog", "saw", k=0))
t = tokens(test)
log_p = sum(math.log(prob(a, b)) for a, b in zip(t, t[1:]))
N = len(t) - 1                             # number of predicted tokens
print("log P =", round(log_p, 3), " perplexity =", round(math.exp(-log_p / N), 2))
```

The output is `unsmoothed P(saw | dog) = 0.0`, then `log P = -10.723  perplexity = 5.97`. Let us check one factor by hand. The vocabulary of possible next words has $V = 9$ entries (*the, cat, sat, on, mat, dog, rug, saw,* `</s>`). The word *dog* occurs twice as a previous word and is never followed by *saw*, so $P(\text{saw} \mid \text{dog}) = (0 + 1)/(2 + 9) = 1/11$. The test sentence has $N = 6$ predictions, and the perplexity 5.97 is clearly below 9, the perplexity of a uniform guess: even this tiny model has learned something.

Two warnings. First, perplexity must always be measured on held-out text; on the training data it is too optimistic. Second, perplexities are comparable only between models with the *same tokenizer*: a model that splits text into more, smaller tokens predicts more, easier tokens, and its per-token perplexity is not directly comparable.

## The limits of counting

N-gram models are fast, simple and interpretable, and with smoothing and very large corpora they worked well for decades. But they have two deep limitations.

1. **Short memory.** The model sees only $n - 1$ previous words. In *"Книга, которую мне вчера подарил брат, оказалась очень ___"*, the right form of the adjective (feminine *интересной*) depends on the word *Книга* at the very beginning.
2. **No similarity.** For an n-gram model, every word is an unrelated symbol. Seeing *"the cat sat"* many times tells it nothing about *"the kitten sat"*. Each word, and each context, must be learned separately.

The second limitation is the key to the next chapter. If words were represented not as isolated symbols but as points in a space, where *cat* and *kitten* are close, knowledge could transfer between similar words. These representations are called embeddings.

## Summary

- Bag of words represents a document by word counts, ignoring order; TF-IDF down-weights words that appear in every document.
- An n-gram model uses the Markov assumption: the next word depends only on the previous $n - 1$ words; probabilities are estimated by relative frequencies (maximum likelihood).
- Unseen n-grams get zero probability; smoothing (add-k, backoff, interpolation, Kneser–Ney) moves some probability to them.
- Perplexity, $\exp$ of the average negative log-probability per token on held-out text, measures model quality; lower is better, and a uniform model over $V$ words has perplexity $V$.
- Counting cannot use long contexts and cannot share knowledge between similar words — the motivation for embeddings and neural models.

## Exercises

1. Explain in your own words why a single unseen bigram makes an unsmoothed bigram model useless for evaluating a sentence.
2. Using the code, compute the perplexity of the test sentence for $k = 0.1$ and $k = 2$. Which value gives the lower perplexity here? Why might a smaller $k$ be better?
3. Compute by hand $P_{\text{add-1}}(\text{sat} \mid \text{cat})$ for the training corpus of this chapter. Compare it with the unsmoothed value $1/2$.
4. Compute the TF-IDF weight of the word *goal* in a document where it occurs 3 times, in a collection of $D = 100$ documents of which 10 contain *goal* (use the natural logarithm).

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| bag of words | мешок слов | 단어 주머니 |
| sparse vector | разреженный вектор | 희소 벡터 |
| TF-IDF | TF-IDF | TF-IDF |
| n-gram | n-грамма | N-그램 |
| unigram / bigram / trigram | униграмма / биграмма / триграмма | 유니그램 / 바이그램 / 트라이그램 |
| Markov assumption | марковское предположение | 마르코프 가정 |
| maximum likelihood estimation | оценка максимального правдоподобия | 최대 가능도 추정 |
| data sparsity | разреженность данных | 데이터 희소성 |
| smoothing | сглаживание | 스무딩(평활화) |
| Laplace smoothing | сглаживание Лапласа | 라플라스 스무딩 |
| backoff | откат | 백오프 |
| interpolation | интерполяция | 보간 |
| test set | тестовая выборка | 테스트 세트 |
| perplexity | перплексия | 퍼플렉서티 |
| cross-entropy | перекрёстная энтропия | 교차 엔트로피 |

---
[← Previous: Tokenization and Subwords (BPE, WordPiece, SentencePiece)](03-tokenization.md) · [Contents](README.md) · [Next: Embeddings: Words as Vectors (Word2Vec, GloVe) →](05-embeddings.md)
