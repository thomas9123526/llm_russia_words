# Chapter 26. Training Data: Collection, Cleaning and Deduplication

> **Part 6 — Training at Scale** · Level: Advanced
>
> **In this chapter you will learn:** where pre-training text comes from and how it is extracted · how heuristic and model-based filters decide what is "good" text · how MinHash and LSH find near-duplicates among billions of documents

Open any web page and look at it as a language model would. Next to the article there is a navigation menu, a cookie banner, a list of "related posts", a copyright line, perhaps a block of keywords for search engines. Somewhere else on the web the same article has been copied, with a different menu around it, a hundred times. If we fed all of this to the model unchanged, a large part of its compute budget would go into learning menus and memorising copies.

In Part 5 we treated the training corpus as a given: "a few trillion tokens of text". This chapter opens that box. The quality of the data is at least as important as the architecture, and most of the work in a modern pre-training project is data work.

## Where the text comes from

A pre-training corpus is usually a **mixture** of several sources:

- **Web crawls.** The largest source by far. The public Common Crawl project has collected snapshots of billions of web pages over many years; most open corpora (C4, The Pile's web part, RefinedWeb, FineWeb and others) start from it.
- **Books and long documents**, which teach long-range coherence.
- **Encyclopaedias** such as Wikipedia: clean, factual, available in many languages.
- **Scientific papers** (for example, arXiv) and **code** (public repositories). Code is now a standard ingredient, because it seems to help with structured reasoning as well as programming.
- **Dialogues and forums**, which contain questions, answers and informal language.

The raw crawl is stored as HTML. The first step is **text extraction**: turning HTML into plain text and removing **boilerplate** — menus, footers, buttons, advertisements. Tools for this use rules about the structure of the page (for example, long paragraphs of text are probably content; short link-heavy blocks are probably navigation). Extraction quality matters more than one would expect: a careless extractor leaves thousands of repeated "Accept cookies" lines in the corpus.

The second step is **language identification**. A small, fast classifier (fastText-style models over character n-grams are typical) predicts the language of each document and a confidence score. This is how a corpus is split into its English, Russian, Korean and other parts — a theme we return to in Chapter 30.

Finally, the sources are combined with chosen weights. This **data mixing** is a design decision: a project may decide, for example, that code makes up a certain fraction of the tokens and that Wikipedia is seen several times while the web is seen once. There is no single correct recipe; mixtures are tuned with small-scale experiments, using the scaling methods of Chapter 27.

## Cleaning and quality filtering

After extraction, most web text is still not something we want the model to imitate. Filtering is done in two main ways.

**Heuristic filters** are simple rules computed for each document. Typical examples, similar to those used for the C4 and Gopher corpora:

- the document has too few words, or is absurdly long;
- the mean word length is outside a normal range;
- too large a fraction of lines end without punctuation, or start with a bullet;
- too large a fraction of characters are symbols such as `#` or `{`;
- many lines or paragraphs are repeated inside the document;
- the document contains almost none of the most common function words of its language ("the", "and", "of" in English) — a sign that it is a list of keywords rather than prose.

Each rule is cheap, transparent and easy to tune. Together they remove a large share of the crawl.

**Model-based filters** learn what good text looks like. One approach trains a classifier to separate a reference set of high-quality text (for example, Wikipedia or pages linked from it) from random web text, and keeps documents that score highly. Another approach, used in CCNet, trains a small n-gram language model on Wikipedia and computes the **perplexity** (Chapter 4) of each web document under it: very high perplexity means the text is very unlike the reference. More recent pipelines also ask an existing LLM to rate the educational value of a sample of pages and then train a cheap classifier to reproduce these ratings on the whole crawl.

Two further filters address harm rather than quality. **Toxicity** filters remove hate speech and similar content, and **PII** filters find and mask personally identifiable information such as e-mail addresses and phone numbers.

Every filter has a cost. A classifier trained on Wikipedia prefers text that looks like Wikipedia, and may discard dialects, informal registers and small languages more often than standard English. Filtering decisions shape what the model knows and whose language it speaks well, so they should be measured, not just assumed to help.

## Deduplication

The web is full of copies: mirrored sites, syndicated news, templates, licence texts, spam farms. Removing them — **deduplication** — has been shown to help in several ways (Lee et al., 2022, "Deduplicating Training Data Makes Language Models Better"):

- compute is not wasted on seeing the same text many times;
- the model memorises less training text verbatim, which matters for privacy and copyright;
- fewer test examples leak into the training set.

**Exact deduplication** is easy: compute a hash (for example, SHA-1) of each document, or of each line or paragraph, and keep one copy per hash. This needs only one pass over the data and a large hash table.

**Near-duplicates** are harder: two pages that differ only in a date, a name or the menu around them have different hashes. To compare them we first turn each document into a set of **shingles** — overlapping word n-grams. For $n = 3$, the text "the cat sat on the mat" gives the set {"the cat sat", "cat sat on", "sat on the", "on the mat"}. The similarity of two documents with shingle sets $A$ and $B$ is the **Jaccard similarity**:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|},
$$

the number of shared shingles divided by the number of distinct shingles in both. $J = 1$ means identical sets, $J = 0$ means nothing in common.

Computing $J$ for every pair of $10^9$ documents is impossible: that is about $5 \cdot 10^{17}$ pairs. **MinHash** solves the first half of the problem. Take a random hash function $h$ and compute the minimum hash value over the shingles of each document, $\min_{x \in A} h(x)$. A short argument shows that

$$
P\left[\min_{x \in A} h(x) = \min_{x \in B} h(x)\right] = J(A, B).
$$

Why? Look at all shingles in $A \cup B$. Under a random hash, each of them is equally likely to have the smallest value. The minima of $A$ and $B$ coincide exactly when that smallest shingle lies in $A \cap B$, which happens with probability $|A \cap B| / |A \cup B|$. With $k$ independent hash functions we get a **signature** of $k$ numbers per document, and the fraction of equal positions in two signatures estimates $J$.

The second half of the problem — avoiding all pairs — is solved by **locality-sensitive hashing (LSH)**. Split the signature of length $k = b \cdot r$ into $b$ bands of $r$ numbers each. Two documents become a *candidate pair* if they agree completely in at least one band. If their Jaccard similarity is $s$, the probability of this is

$$
P(\text{candidate}) = 1 - (1 - s^r)^b .
$$

Here $s^r$ is the probability of agreeing on one whole band, and $(1 - s^r)^b$ is the probability of failing in all $b$ bands. With $b = 20$, $r = 5$ (100 hashes), a pair with $s = 0.8$ becomes a candidate with probability about $0.9996$, a pair with $s = 0.5$ with probability about $0.47$, and a pair with $s = 0.3$ only about $0.05$. The curve is a steep "S": similar pairs are almost always found, dissimilar pairs almost never. In practice each band is hashed into a bucket, and only documents that share a bucket are compared.

## MinHash in a few lines

The code below builds 3-word shingles, computes the exact Jaccard similarity and its MinHash estimate. For simplicity, the $k$ "hash functions" are CRC32 applied to the shingle with a different prefix; real systems use faster and better-mixed hash families.

```python
import zlib

def shingles(text, n=3):
    """Set of word n-grams (shingles) of a document."""
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

def jaccard(a, b):
    return len(a & b) / len(a | b)

def minhash(s, num_hashes=128):
    """MinHash signature: the minimum of each of num_hashes hash functions."""
    return [min(zlib.crc32(f"{i}:{x}".encode()) for x in s)
            for i in range(num_hashes)]

def estimate(sig_a, sig_b):
    # Fraction of matching positions ≈ Jaccard similarity
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)

doc1 = "the cat sat on the mat and looked out of the window at the rain"
doc2 = "the cat sat on the mat and looked out of the window at the snow"
doc3 = "gradient descent moves the weights against the gradient of the loss"

s1, s2, s3 = shingles(doc1), shingles(doc2), shingles(doc3)
m1, m2, m3 = minhash(s1), minhash(s2), minhash(s3)
print(f"doc1~doc2: exact {jaccard(s1, s2):.2f}, minhash {estimate(m1, m2):.2f}")
print(f"doc1~doc3: exact {jaccard(s1, s3):.2f}, minhash {estimate(m1, m3):.2f}")
```

It prints `exact 0.86, minhash 0.82` for the first pair and `0.00, 0.00` for the second. The first two documents have 14 distinct shingles in total and share 12 of them, so $J = 12/14 \approx 0.86$; the estimate from 128 hashes is close. A typical pipeline would treat pairs with estimated $J$ above a threshold such as 0.8 as duplicates and keep one document from each group.

## Contamination and documentation

One more check is essential for research: **benchmark contamination**. If the questions of a test set (Chapter 39) appear in the training data, a high score may only show memorisation. A common defence is to search the corpus for long n-grams (for example, 13-grams) that also occur in the test sets and remove the matching documents, and to report which benchmarks were decontaminated.

Finally, a corpus is a research artefact and should be documented: its sources, licences, filters, the share of each language and the known gaps. Formats such as "datasheets for datasets" describe this information. For your PhD, keep in mind that a reproducible description of the data is as important as a description of the model; many results that look like architecture effects are really data effects.

## Summary

- Pre-training corpora are mixtures of web crawls, books, encyclopaedias, papers, code and dialogue; the mixture weights are a tuned design choice.
- Raw HTML must be converted to text with boilerplate removal, and each document is labelled with its language.
- Heuristic filters (length, symbols, repetition, stop words) and model-based filters (quality classifiers, perplexity) remove low-quality text, but they can also remove dialects and small languages.
- Exact duplicates are removed with hashes; near-duplicates with shingles, Jaccard similarity, MinHash signatures and LSH banding.
- Deduplication saves compute, reduces memorisation and limits test-set leakage; decontamination and documentation make results trustworthy.

## Exercises

1. Explain in your own words why the probability that two MinHash values coincide equals the Jaccard similarity.
2. With $b = 10$ bands of $r = 10$ rows, compute the probability that pairs with $s = 0.5$ and $s = 0.9$ become candidates. Compare with the values for $b = 20$, $r = 5$ in the text. Which setting is stricter?
3. Modify the code to use 5-word shingles. How do the exact and estimated similarities of `doc1` and `doc2` change, and why?
4. Write three heuristic filter rules that would make sense for Russian or Korean web text but not for English. What could go wrong with each?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| corpus | корпус | 말뭉치 |
| web crawling | обход веб-страниц | 웹 크롤링 |
| text extraction | извлечение текста | 텍스트 추출 |
| boilerplate removal | удаление шаблонного текста | 상용구 제거 |
| language identification | определение языка | 언어 식별 |
| data mixing | смешивание данных | 데이터 혼합 |
| quality filtering | фильтрация по качеству | 품질 필터링 |
| heuristic filter | эвристический фильтр | 휴리스틱 필터 |
| personally identifiable information (PII) | персональные данные | 개인 식별 정보 |
| deduplication | дедупликация | 중복 제거 |
| near-duplicate | нечёткий дубликат | 유사 중복 |
| Jaccard similarity | коэффициент Жаккара | 자카드 유사도 |
| MinHash | MinHash | 민해시 |
| locality-sensitive hashing (LSH) | локально-чувствительное хеширование | 지역 민감 해싱 |
| benchmark contamination | загрязнение бенчмарка | 벤치마크 오염 |

---
[← Previous: Modern Open Models: LLaMA, GQA, Mistral, Qwen](25-modern-open-models.md) · [Contents](README.md) · [Next: Scaling Laws and Compute-Optimal Training →](27-scaling-laws.md)
