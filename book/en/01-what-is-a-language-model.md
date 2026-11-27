# Chapter 1. What Is a Language Model?

> **Part 1 — Text and Representation** · Level: Beginner
>
> **In this chapter you will learn:** what it means to predict the next word · how the probability of a whole sentence is built from small pieces · how a model generates text one token at a time

## A game everyone can play

Read this unfinished sentence: *"I poured milk into my morning ___"*. Most people say *coffee* or *tea* at once. Few people say *car*, and nobody says *the*. You did not look anything up. You used years of experience with English to decide which continuations are likely and which are not.

This small game is the whole idea of a **language model**. A language model is a system that, given some text, assigns a probability to every possible continuation. For the sentence above, a good model might say: *coffee* 0.45, *tea* 0.30, *cereal* 0.10, and so on, with tiny probabilities for strange words. The numbers must be non-negative and must add up to 1, because exactly one word comes next.

It may seem surprising that such a simple task is the basis of modern systems that translate, summarize, write code and answer questions. The reason is that guessing the next word well is hard. To continue *"The capital of France is ___"* you need knowledge of the world. To continue *"If x = 3, then 2x + 1 = ___"* you need some arithmetic. To continue a Russian sentence correctly you need to choose the right grammatical case. A model that predicts well must, in some form, learn all of these things. This book is about how such models are built, from the simplest counting methods to large Transformer networks.

## Text as a sequence of tokens

Before we can talk about probabilities, we must decide what the "pieces" of text are. A model does not see letters on a page; it sees a sequence of symbols called **tokens**. A token can be a whole word, a part of a word, a single character or even a single byte. For now, think of tokens as words. In Chapter 3 we will see why modern models use pieces of words (subwords) instead.

The set of all tokens a model knows is its **vocabulary**. If the vocabulary has $V$ tokens, then the model's prediction at each step is a list of $V$ numbers, one probability per token. Such a list is called a **probability distribution** over the vocabulary.

We write a text of $N$ tokens as $w_1, w_2, \dots, w_N$. Here $w_1$ is the first token, $w_2$ the second, and $w_N$ the last. The tokens that come before position $t$ are called the **context** of that position and are written $w_{<t}$, which is short for $w_1, \dots, w_{t-1}$.

The central object of this book is the **conditional probability**

$$P(w_t \mid w_{<t}),$$

read as "the probability of token $w_t$ given all previous tokens". The vertical bar $\mid$ means "given". For our example, $P(\text{coffee} \mid \text{I poured milk into my morning})$ is high and $P(\text{car} \mid \text{I poured milk into my morning})$ is low. This task is called **next-token prediction**.

## The probability of a whole sentence

A language model can do more than guess one word. It can also tell us how likely a whole sentence is. This is useful: a speech recognizer can choose between *"recognize speech"* and *"wreck a nice beach"*, which sound almost the same, by asking which sentence is more probable.

How do we get the probability of a whole sequence from next-token predictions? Probability theory gives us the **chain rule**:

$$P(w_1, w_2, \dots, w_N) = \prod_{t=1}^{N} P(w_t \mid w_{<t}).$$

The symbol $\prod$ means "multiply all the terms": for $t = 1$ we take $P(w_1)$, for $t = 2$ we take $P(w_2 \mid w_1)$, and so on. In words: the probability of a sentence is the probability of the first word, times the probability of the second word given the first, times the probability of the third given the first two, and so on until the end. No approximation is involved here; this is an exact identity. All the difficulty lies in estimating each factor $P(w_t \mid w_{<t})$.

The chain rule tells us something important. If we have a good next-token predictor, we automatically have a model of whole texts. Learning a language model therefore reduces to learning one thing well: predicting the next token from its context.

Let us see a tiny example. Suppose our whole "world" is three sentences:

- *the cat sat on the mat*
- *the dog sat on the rug*
- *the cat saw the dog*

We add a special start symbol `<s>` and an end symbol `</s>` to each sentence, so that the model can also predict how sentences begin and end. Now we estimate each probability by counting. The word *the* appears 6 times as a previous word, and it is followed by *cat* twice, *dog* twice, *mat* once and *rug* once. So $P(\text{cat} \mid \text{the}) = 2/6 = 1/3$. If we look only at the one previous word, as here, the model is called a **bigram** model (we will study it properly in Chapter 4). With this simplification, the probability of *"the cat sat on the mat"* is

$$P(\text{the} \mid \texttt{<s>}) \cdot P(\text{cat} \mid \text{the}) \cdot P(\text{sat} \mid \text{cat}) \cdot P(\text{on} \mid \text{sat}) \cdot P(\text{the} \mid \text{on}) \cdot P(\text{mat} \mid \text{the}) \cdot P(\texttt{</s>} \mid \text{mat})$$

$$= 1 \cdot \tfrac{1}{3} \cdot \tfrac{1}{2} \cdot 1 \cdot 1 \cdot \tfrac{1}{6} \cdot 1 = \tfrac{1}{36}.$$

Every sentence in this toy world starts with *the*, so $P(\text{the} \mid \texttt{<s>}) = 1$. The word *cat* is followed once by *sat* and once by *saw*, so $P(\text{sat} \mid \text{cat}) = 1/2$.

## Generating text

A language model can also produce new text. The procedure is simple:

1. Start with a context (it may be just the start symbol, or a question from a user).
2. Ask the model for the distribution over the next token.
3. Choose one token from this distribution, for example by random **sampling** according to the probabilities.
4. Append the chosen token to the context and go back to step 2.
5. Stop when the model produces the end symbol or when the text is long enough.

Because each new token depends on the tokens produced before it, this is called **autoregressive generation** ("auto" = self, "regressive" = depending on previous values). Every chatbot you have used works in this way: it writes its answer one token at a time. In Chapter 24 we will study different ways to choose the next token (greedy choice, temperature, top-k, top-p).

Here is the whole idea in a few lines of Python. The program counts bigrams in our three sentences and then samples a new sentence.

```python
import random
from collections import defaultdict, Counter

corpus = ["the cat sat on the mat", "the dog sat on the rug", "the cat saw the dog"]

# Count which word follows which; <s> and </s> mark sentence start and end
counts = defaultdict(Counter)
for sentence in corpus:
    words = ["<s>"] + sentence.split() + ["</s>"]
    for prev, nxt in zip(words, words[1:]):
        counts[prev][nxt] += 1

def next_word_probs(prev):
    total = sum(counts[prev].values())
    return {w: c / total for w, c in counts[prev].items()}

print(next_word_probs("the"))  # P(next word | "the")

# Generate a sentence one word at a time
random.seed(0)
word, output = "<s>", []
while True:
    probs = next_word_probs(word)
    word = random.choices(list(probs), weights=list(probs.values()))[0]
    if word == "</s>":
        break
    output.append(word)
print(" ".join(output))
```

The first line of output is `{'cat': 0.333..., 'mat': 0.166..., 'dog': 0.333..., 'rug': 0.166...}`. The second line is `the dog sat on the mat` — a sentence that does not appear in our data. Even this tiny model combines pieces of what it has seen into something new. Of course, it can also produce nonsense such as *the cat saw the rug*, because it looks only one word back.

## From counting to large neural models

The idea of a language model is old. Claude Shannon, the founder of information theory, already experimented in the 1940s with predicting English letters and words from the previous ones. For several decades the main tool was the **n-gram model**: estimate $P(w_t \mid w_{<t})$ by counting how often short word sequences occur in a large collection of texts, called a **corpus**. N-gram models were used in speech recognition and machine translation for many years.

Counting has a basic weakness: most long word sequences never appear in any corpus, so their counts are zero, and the model knows nothing about them. The next step was to replace counting with a **neural network**. In 2003, Bengio and colleagues proposed a neural language model that represents each word as a vector of numbers (an *embedding*, Chapter 5) and computes the next-word distribution with a network. Similar words get similar vectors, so knowledge transfers from one word to another.

Later came recurrent networks (Chapters 11–12), which read text one token at a time and keep a memory of the past, and then, in 2017, the **Transformer** (Chapters 15–20), which uses a mechanism called *attention* to look at all previous tokens at once. Modern **large language models** (LLMs) are Transformers with billions of **parameters** — numbers learned from data — trained on huge corpora to predict the next token.

It is worth stressing what has *not* changed. A large modern model and our ten-line bigram program solve the same task: they estimate $P(w_t \mid w_{<t})$. The differences are in how the context is represented, how much of it is used, and how the probabilities are computed. Here is the road map of this book:

- **Part 1** (this part): how text becomes numbers — characters, tokens, counts and embeddings.
- **Part 2**: the neural network tools we need — vectors, layers, loss functions, gradient descent.
- **Parts 3–4**: models of sequences, attention and the Transformer.
- **Parts 5–8**: real language models, how they are trained at scale, adapted, aligned and used.

## Summary

- A language model assigns probabilities to possible continuations of a text; at each step it outputs a probability distribution over its vocabulary.
- Text is treated as a sequence of tokens $w_1, \dots, w_N$; the central quantity is the conditional probability $P(w_t \mid w_{<t})$.
- The chain rule turns next-token probabilities into the probability of a whole sequence, exactly.
- Text generation is autoregressive: predict a distribution, choose a token, append it, repeat.
- From n-gram counts to Transformers, the task stays the same; only the way of estimating the probabilities changes.

## Exercises

1. Explain in your own words why predicting the next word well requires knowledge of grammar and of the world. Give one example in Russian or Korean.
2. Using the three-sentence corpus from this chapter, compute by hand the bigram probability of *"the dog sat on the rug"*. Compare it with the probability of *"the cat sat on the mat"*. What do you notice, and why?
3. Run the code with different values of `random.seed`. Which sentences can the model produce that are not in the corpus? Can it produce a sentence of infinite length? Explain.
4. Write the chain rule for the sentence *"я люблю чай"* (three tokens plus `</s>`) as a product of four conditional probabilities.

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| language model | языковая модель | 언어 모델 |
| large language model (LLM) | большая языковая модель | 대규모 언어 모델 |
| token | токен | 토큰 |
| vocabulary | словарь | 어휘 집합 |
| probability distribution | распределение вероятностей | 확률 분포 |
| conditional probability | условная вероятность | 조건부 확률 |
| context | контекст | 문맥 |
| next-token prediction | предсказание следующего токена | 다음 토큰 예측 |
| chain rule | цепное правило | 연쇄 법칙 |
| bigram | биграмма | 바이그램 |
| sampling | сэмплирование | 샘플링 |
| autoregressive generation | авторегрессионная генерация | 자기회귀 생성 |
| corpus | корпус текстов | 말뭉치 |
| neural network | нейронная сеть | 신경망 |
| parameter | параметр | 매개변수 |

---
[Contents](README.md) · [Next: Text as Data: Characters, Words and Encodings →](02-text-as-data.md)
