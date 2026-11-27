# Chapter 3. Tokenization and Subwords (BPE, WordPiece, SentencePiece)

> **Part 1 — Text and Representation** · Level: Beginner
>
> **In this chapter you will learn:** why modern models split words into subword pieces · how Byte Pair Encoding learns its vocabulary step by step · how WordPiece, the Unigram model and SentencePiece differ from BPE

## A word the model has never seen

Suppose a model was trained on a large Russian corpus, and a user types the word *неперезагружаемый* ("not reloadable"). Perhaps this exact form never appeared in the training data. What should the model do with it?

In Chapter 2 we saw that "word" is a slippery notion, especially in Russian and Korean. We have three basic options for the units of text:

- **Words.** Each word form is one token. Sequences are short, and each token carries a lot of meaning. But the vocabulary must be enormous to cover all forms (*кот, кота, коту, …*), and any form not in the vocabulary becomes an **out-of-vocabulary word** (OOV). It is usually replaced by a special **unknown token** `[UNK]`, and all its information is lost.
- **Characters.** The vocabulary is small (a few hundred symbols for a few scripts), and nothing is ever unknown. But sequences become long — a sentence of 10 words may be 60 characters — and each character carries little meaning by itself. Long sequences cost more computation, as we will see when we study attention.
- **Subwords.** Frequent words stay whole; rare words are split into frequent pieces. Our word might become *не + пере + загружа + емый*. The vocabulary has a moderate size (typically tens of thousands of tokens), sequences are of moderate length, and there are no unknown words, because in the worst case we can fall back to single characters.

Almost all modern language models use subwords. The question is how to choose the pieces. We do not want to write rules for every language by hand. Instead, the pieces are *learned from data* by a **tokenizer** training algorithm. We study the three most important ones.

## Byte Pair Encoding

**Byte Pair Encoding** (BPE) was originally a data compression method. Sennrich, Haddow and Birch adapted it for neural machine translation in 2016. The idea is simple: start from characters and repeatedly merge the most frequent adjacent pair of symbols into a new symbol.

The training algorithm is:

1. Split every word in the corpus into characters, and add an end-of-word marker `</w>` so that we can tell *est* at the end of a word from *est* in the middle.
2. Count all pairs of adjacent symbols, weighting each pair by the frequency of the word it appears in.
3. Merge the most frequent pair everywhere into one new symbol, and record this **merge rule**.
4. Repeat steps 2–3 until the vocabulary reaches the desired size.

Here is a complete implementation on the classic toy example from the BPE paper: the word *low* appears 5 times, *lower* 2 times, *newest* 6 times and *widest* 3 times.

```python
from collections import Counter
def get_pair_counts(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq          # each pair counts as often as its word
    return pairs

def merge_pair(pair, vocab):
    new_vocab = {}
    for word, freq in vocab.items():
        symbols, out, i = word.split(), [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])   # merge the pair
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_vocab[" ".join(out)] = freq
    return new_vocab

vocab = {"l o w </w>": 5, "l o w e r </w>": 2,          # word counts; words are split
         "n e w e s t </w>": 6, "w i d e s t </w>": 3}  # into characters, </w> = end
for step in range(1, 6):
    pairs = get_pair_counts(vocab)
    best = max(pairs, key=pairs.get)       # the most frequent pair
    vocab = merge_pair(best, vocab)
    print(step, best, list(vocab))
```

The five merges it learns are:

| Step | Merged pair | Frequency | Words after the merge |
|---|---|---|---|
| 1 | `e` + `s` | 6 + 3 = 9 | `n e w es t </w>`, `w i d es t </w>` |
| 2 | `es` + `t` | 9 | `n e w est </w>`, `w i d est </w>` |
| 3 | `est` + `</w>` | 9 | `n e w est</w>`, `w i d est</w>` |
| 4 | `l` + `o` | 5 + 2 = 7 | `lo w </w>`, `lo w e r </w>` |
| 5 | `lo` + `w` | 7 | `low </w>`, `low e r </w>` |

Notice what happened: without any knowledge of English, the algorithm discovered the suffix *-est* and the stem *low*. (At step 1 the pair `s t` also had frequency 9; when there is a tie, our code takes the pair it saw first.)

To **tokenize** a new word, we split it into characters and apply the learned merges *in the order they were learned*. The word *lowest* never appeared in training, but it becomes `l o w e s t </w>` → … → `low est</w>`: two meaningful pieces. A word with unfamiliar patterns simply stays split into smaller pieces; it never becomes unknown.

The number of merges is a **hyperparameter** — a setting chosen by the designer, not learned. The final vocabulary contains the base characters plus one new symbol per merge. More merges give a larger vocabulary and shorter sequences.

### Byte-level BPE

A plain character-level BPE still has a problem: the base alphabet must contain every character that may ever appear, and Unicode has more than a hundred thousand of them. GPT-2 (2019) solved this with **byte-level BPE**: the text is first encoded in UTF-8 (Chapter 2), and BPE starts from the 256 possible byte values. Every possible text can be represented, so there is no unknown token at all. Frequent byte sequences, such as the two bytes of a Cyrillic letter or the three bytes of a common Hangul syllable, are quickly merged into single tokens. Byte-level BPE is used by the GPT family and many other models.

## WordPiece

**WordPiece** was developed at Google and became well known as the tokenizer of BERT (Chapter 22). It works like BPE — start from characters, merge pairs — but chooses the pair differently. Instead of the most frequent pair, it chooses the pair with the highest score

$$\text{score}(a, b) = \frac{\text{count}(ab)}{\text{count}(a) \cdot \text{count}(b)},$$

where $\text{count}(ab)$ is how often $a$ is directly followed by $b$, and $\text{count}(a)$, $\text{count}(b)$ are the frequencies of each symbol separately. This prefers pairs whose parts occur together *more often than expected* from their individual frequencies. Two very common symbols that happen to stand next to each other get a lower score than two rarer symbols that almost always appear together.

WordPiece also marks pieces differently: a piece that continues a word gets the prefix `##`. For example, BERT may tokenize *tokenization* as `token`, `##ization`. To tokenize a new word, WordPiece uses a greedy **longest-match-first** rule: take the longest vocabulary entry that matches the beginning of the word, then continue from where it ended.

## The Unigram model and SentencePiece

BPE and WordPiece build the vocabulary from the bottom up, by merging. The **Unigram language model** tokenizer, proposed by Kudo in 2018, works the other way: it starts with a large set of candidate pieces and removes pieces until the vocabulary is small enough.

The Unigram model gives each piece $x$ a probability $p(x)$, and assumes that pieces are independent. The probability of one particular segmentation $\mathbf{x} = (x_1, \dots, x_M)$ of a text is then

$$P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i).$$

A word usually has many possible segmentations (*котами* = *кот + ами* or *ко + та + ми* or …). The tokenizer chooses the one with the highest probability; this can be found efficiently with a dynamic programming method called the **Viterbi algorithm**. During training, the probabilities $p(x)$ are estimated on the corpus, and at each round the pieces whose removal hurts the total likelihood of the corpus least are removed. Single characters are always kept, so every text can still be tokenized.

Because the Unigram model is probabilistic, it can also *sample* different segmentations of the same word during training. This trick, called **subword regularization**, makes the model more robust to variations in spelling and segmentation.

**SentencePiece** (Kudo and Richardson, 2018) is a library rather than a new algorithm. It implements both BPE and the Unigram model, with two important design choices:

- It treats the input as a raw stream of characters, *including spaces*. It does not need a separate step that first splits text into words, so it works the same way for English, Russian, Korean, Japanese and Chinese.
- It replaces the space with a visible symbol `▁` (U+2581). The sentence *"Я люблю чай"* might become `▁Я`, `▁люб`, `лю`, `▁чай`. Because spaces are kept as part of tokens, the original text can be restored exactly from the tokens.

SentencePiece is used by many multilingual models, for example T5 and the first LLaMA models.

## Tokenizers in practice

A real tokenizer contains more than the learned pieces. It has **special tokens** with fixed roles, for example a beginning-of-sequence token, an end-of-sequence token (like our `<s>` and `</s>` in Chapter 1), a **padding** token used to make sequences in a batch equal in length, and, in chat models, tokens that mark the start and end of each message.

The tokenizer is trained once, before the model, and then frozen; the model can only ever see text through it. Its choices have visible effects:

- **Different languages cost different amounts.** A tokenizer trained mostly on English text has many English words as single tokens, but splits Russian and Korean words into more pieces. The average number of tokens per word is sometimes called the tokenizer's **fertility**. Higher fertility means longer sequences, slower and more expensive processing, and less text fitting into the model's context window. We return to this in Chapter 30.
- **Numbers and spelling.** A number such as *12345* may be split into irregular pieces like *123* + *45*, which makes arithmetic harder to learn. Similarly, a model that sees *strawberry* as two or three pieces does not directly "see" its letters.
- **Vocabulary size is a trade-off.** A larger vocabulary gives shorter sequences but more parameters in the embedding layer (Chapter 5), and rare tokens are seen too few times to learn them well.

## Summary

- Word-level vocabularies suffer from unknown words; character-level ones produce long sequences; subword tokenization is the compromise used by modern models.
- BPE starts from characters (or bytes) and repeatedly merges the most frequent adjacent pair; new words are tokenized by replaying the merges in order.
- Byte-level BPE starts from 256 byte values, so no text is ever unknown.
- WordPiece merges the pair with the highest $\text{count}(ab) / (\text{count}(a)\,\text{count}(b))$ and marks continuation pieces with `##`.
- The Unigram model starts from many pieces, prunes them, and picks the most probable segmentation; SentencePiece implements BPE and Unigram on raw text with `▁` for spaces.

## Exercises

1. Explain in your own words why a subword vocabulary has no out-of-vocabulary words, while a word vocabulary does.
2. Continue the BPE example by hand for two more steps. Which pairs are merged at steps 6 and 7, and with what frequencies? Check your answer by changing `range(1, 6)` to `range(1, 8)`.
3. Using the five merges from the table, tokenize the words *lowest*, *newer* and *wider*. Which of them get meaningful pieces, and why?
4. Replace the toy vocabulary with Russian words, for example `{"к о т </w>": 5, "к о т а </w>": 3, "к о т у </w>": 2, "к о т о м </w>": 2}`. Which merges does BPE learn first?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| tokenizer | токенизатор | 토크나이저 |
| subword tokenization | токенизация на подслова | 서브워드 토큰화 |
| out-of-vocabulary word (OOV) | слово вне словаря | 어휘 집합 밖 단어(OOV) |
| unknown token | неизвестный токен | 미지 토큰 |
| Byte Pair Encoding (BPE) | кодирование пар байтов (BPE) | 바이트 쌍 인코딩(BPE) |
| merge rule | правило слияния | 병합 규칙 |
| byte-level BPE | BPE на уровне байтов | 바이트 수준 BPE |
| WordPiece | WordPiece | WordPiece |
| Unigram language model | униграммная языковая модель | 유니그램 언어 모델 |
| SentencePiece | SentencePiece | SentencePiece |
| subword regularization | регуляризация подслов | 서브워드 정규화 |
| special token | специальный токен | 특수 토큰 |
| padding | дополнение | 패딩 |
| tokenizer fertility | фертильность токенизатора | 토크나이저 분할률(fertility) |
| hyperparameter | гиперпараметр | 하이퍼파라미터 |

---
[← Previous: Text as Data: Characters, Words and Encodings](02-text-as-data.md) · [Contents](README.md) · [Next: Counting Words: Bag of Words, N-grams and Perplexity →](04-counting-words.md)
