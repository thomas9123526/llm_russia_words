# Chapter 30. Multilingual Models and Cross-Lingual Transfer

> **Part 6 — Training at Scale** · Level: Advanced
>
> **In this chapter you will learn:** how one model learns many languages from a shared vocabulary and a mixed corpus · why Russian and Korean text is more expensive to tokenize than English, and how temperature sampling balances languages · what cross-lingual transfer is, why it works, and where it breaks down

Here is an experiment that surprised researchers when it was first done. Take a multilingual model, fine-tune it to classify the sentiment of movie reviews using **English examples only**, and then give it two reviews it has never seen in a language it was never fine-tuned on:

- «Фильм был потрясающим, я смотрел его два раза.» (Russian: "The film was amazing, I watched it twice.")
- «영화가 너무 지루해서 중간에 나왔다.» (Korean: "The film was so boring that I left in the middle.")

The model labels the first as positive and the second as negative far more often than chance would allow. No Russian or Korean labels were used. This is **cross-lingual transfer**, and it is the central phenomenon of this chapter. For you, as a researcher who reads in Korean, English and Russian, it is also a practical question: how well do LLMs actually work in your languages, and why?

## One model, many languages

A **multilingual model** is trained in the usual way — by masked or causal language modeling (Chapter 21) — on a corpus that mixes many languages, with one shared tokenizer and vocabulary. No dictionary or parallel text is required.

The first widely studied example was **multilingual BERT (mBERT)**: the BERT architecture of Chapter 22, trained on Wikipedia in about a hundred languages with a shared WordPiece vocabulary. **XLM** (Lample and Conneau, 2019) added an objective on parallel sentence pairs, *translation language modeling*, in which a masked word may be predicted from the context in either language. **XLM-R** (Conneau et al., 2020) went back to plain masked language modeling but used much more data: filtered Common Crawl text in 100 languages, prepared with the kind of pipeline described in Chapter 26, and a SentencePiece vocabulary (Chapter 3) of 250,000 tokens. Encoder–decoder models such as mT5 and many decoder-only LLMs are trained multilingually in the same spirit.

The data is very unbalanced. On the web, English is by far the largest language; Russian is among the larger ones; Korean has much less text; and many of the world's languages have almost none. In the terms of the field, there are **high-resource** and **low-resource** languages, and the gap between them is several orders of magnitude.

## Tokenization across scripts

The first place where languages are treated unequally is the tokenizer. Different languages use different **writing systems** (scripts). Recall from Chapter 2 that UTF-8 uses 1 byte for Latin letters, 2 bytes for Cyrillic letters and 3 bytes for Hangul syllables. A byte-level tokenizer trained mostly on English has learned many merges for English and relatively few for other scripts.

```python
import numpy as np

sentences = {
    "en": "The model learns language from data.",
    "ru": "Модель изучает язык по данным.",
    "ko": "모델은 데이터로부터 언어를 배운다.",
}
for lang, s in sentences.items():
    n_bytes = len(s.encode("utf-8"))
    print(f"{lang}: {len(s.split())} words, {len(s)} chars, {n_bytes} UTF-8 bytes, "
          f"{n_bytes / len(s):.2f} bytes/char")

# Temperature sampling of languages: p_i ∝ (n_i / N) ** alpha
sizes = np.array([1000.0, 100.0, 20.0])     # data size, e.g. GB: en, ru, ko
q = sizes / sizes.sum()                     # natural proportions
for alpha in [1.0, 0.5, 0.3]:
    p = q**alpha / (q**alpha).sum()
    print(f"alpha={alpha}: en={p[0]:.2f}, ru={p[1]:.2f}, ko={p[2]:.2f}")
```

The first part prints:

```text
en: 6 words, 36 chars, 36 UTF-8 bytes, 1.00 bytes/char
ru: 5 words, 30 chars, 55 UTF-8 bytes, 1.83 bytes/char
ko: 4 words, 19 chars, 49 UTF-8 bytes, 2.58 bytes/char
```

The same meaning costs 36 bytes in English but 55 in Russian and 49 in Korean. (The Russian figure is below 2 bytes per character because spaces and the full stop take one byte each.) A tokenizer that has not learned good merges for Cyrillic and Hangul will turn these bytes into many more tokens.

The standard measure is **tokenizer fertility**: the average number of tokens per word. Higher fertility has three costs. The same text uses more compute; the same context window (Chapter 17) holds less text; and the model must assemble meaning from more, smaller pieces, which usually hurts quality.

Morphology makes the problem larger. Russian is a **morphologically rich language**: one noun has many forms — «книга, книги, книге, книгу, книгой, о книге» ("book" in different cases). Korean is **agglutinative**: particles and endings attach to the stem — «학교», «학교에», «학교에서», «학교에서는», «학교에서만» ("school", "to school", "at school", "at school (topic)", "only at school"). Space-separated "words" are therefore long in both languages, and a good tokenizer should split them into meaningful pieces (stem + ending, or stem + particle) rather than arbitrary ones. Korean adds one more choice: Hangul syllable blocks are built from letters (jamo), so «한» can be kept as one character or decomposed into ㅎ, ㅏ, ㄴ. How many vocabulary entries each language receives — **vocabulary allocation** — is a real design decision for a multilingual model.

## Balancing languages in training

If we sample training documents in proportion to the available data, the model sees almost only English. If we sample all languages equally, the small languages are repeated many times and overfit (Chapter 27, data-constrained scaling). The usual compromise is **temperature sampling**. Let $q_i = n_i / \sum_j n_j$ be the natural share of language $i$, where $n_i$ is its amount of data. We sample language $i$ with probability

$$
p_i = \frac{q_i^{\alpha}}{\sum_j q_j^{\alpha}}, \qquad 0 < \alpha \le 1 .
$$

With $\alpha = 1$ we keep the natural proportions; as $\alpha \to 0$ the distribution becomes uniform. The second part of the code shows the effect for an imaginary corpus of 1000, 100 and 20 units of English, Russian and Korean:

```text
alpha=1.0: en=0.89, ru=0.09, ko=0.02
alpha=0.5: en=0.69, ru=0.22, ko=0.10
alpha=0.3: en=0.55, ru=0.28, ko=0.17
```

At $\alpha = 0.3$ Korean is sampled about 9.6 times as often as its natural share ($0.17$ instead of $20/1120 \approx 0.018$). Values of $\alpha$ between about 0.3 and 0.7 have been used in well-known multilingual models.

Why not simply add more and more languages? Conneau et al. (2020) described the **curse of multilinguality**: with a fixed model size, adding languages at first helps the low-resource ones, through transfer, but beyond some point the performance of all languages starts to fall, because they compete for the same limited capacity. This competition is called negative interference. The remedy is more capacity — a larger model, or more parameters per language, for example language-specific experts in an MoE model (Chapter 29).

## Cross-lingual transfer

Let us return to the experiment from the beginning. **Zero-shot cross-lingual transfer** means: fine-tune on a task in one language (usually English, because labels are cheap there) and evaluate directly in other languages. Benchmarks such as XNLI (natural language inference in 15 languages) were built to measure it.

Why does it work? Several factors contribute:

- **Shared tokens.** Numbers, names, URLs, code and many borrowed words look similar across languages: «компьютер», «컴퓨터» and "computer" are the same word in three scripts; «интернет» and «인터넷» likewise. Early explanations of mBERT's success pointed to such shared pieces. For the same reason, **transliteration** into a common script (for example, writing Russian in Latin letters) has been used to increase the overlap between languages.
- **Shared structure.** Later studies found that transfer happens even between languages with no shared subwords at all, and that model depth matters more. The model seems to discover that different languages express similar meanings in similar contexts, and it reuses the same internal features.
- **Aligned representations.** If we compute the hidden vectors of a sentence and of its translation, they are often close to each other, especially in the middle layers. In effect, the model develops a partly language-neutral space of meaning, and a classifier trained on top of English vectors also works on Russian and Korean ones.

Transfer is not equal for all language pairs. It is usually best between related languages that share a script and word order, and weaker between distant ones. Russian and Korean are distant from English in different ways: Russian uses another script, has free word order and rich case morphology; Korean uses another script, places the verb at the end (subject–object–verb order) and expresses grammar with particles. When zero-shot transfer is not enough, common alternatives are *translate-train* (machine-translate the English training data into the target language) and *translate-test* (translate the test input into English). Even a small number of labelled target-language examples usually helps a lot.

Real users also mix languages in one message — **code-switching**, e.g. «Мне нужно сделать fine-tuning модели» or «이 모델은 inference가 느려요». Multilingual models handle this better than monolingual ones, but it remains a source of errors.

For your research, two more points are useful. First, language-specific models and benchmarks exist for both of your languages: for example, Russian SuperGLUE for Russian and KLUE for Korean, and there are BERT- and GPT-style models trained mainly on Russian or Korean text. Comparing them with multilingual models is a classic research design. Second, a common way to improve an English-centric open model for Russian or Korean is **continued pre-training** on target-language text, sometimes with an extended vocabulary to reduce fertility; we will see related adaptation methods in Part 7. Finally, remember **cultural bias**: a model that speaks Korean fluently may still reason with assumptions taken from English-language sources.

## Summary

- A multilingual model is trained by ordinary language modeling on a mixed corpus with one shared vocabulary (mBERT, XLM, XLM-R, mT5 and multilingual LLMs); parallel data is helpful but not required.
- UTF-8 and English-centric tokenizers make Russian (Cyrillic, rich inflection) and Korean (Hangul, agglutination) more expensive; tokenizer fertility measures this cost.
- Temperature sampling $p_i \propto q_i^\alpha$ raises the share of low-resource languages; the curse of multilinguality shows that languages compete for fixed capacity.
- Zero-shot cross-lingual transfer works because of shared tokens, shared structure and partly aligned representations; it is weaker for typologically distant languages.
- Translate-train, translate-test, a few target-language labels, and continued pre-training with vocabulary extension are practical ways to improve results for a given language.

## Exercises

1. Explain in your own words why a sentiment classifier fine-tuned only on English data can work on Russian and Korean reviews.
2. Using the formula for $p_i$, compute the sampling probabilities for data sizes of 500 (English), 50 (Russian) and 5 (Korean) with $\alpha = 0.5$. How many times more often is Korean sampled than its natural share?
3. Extend the code with three more sentences of your own in each language (for example, from this chapter's Russian and Korean versions). Compute the average bytes per character and per word. Which language has the longest "words" in bytes, and why?
4. Write the six case forms of a Russian noun of your choice and the Korean forms of «학교» with five different particles. How would you want an ideal tokenizer to split them?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| multilingual model | многоязычная модель | 다국어 모델 |
| cross-lingual transfer | межъязыковой перенос | 교차 언어 전이 |
| zero-shot cross-lingual transfer | межъязыковой перенос без примеров | 제로샷 교차 언어 전이 |
| cross-lingual alignment | межъязыковое выравнивание | 교차 언어 정렬 |
| high-resource language | высокоресурсный язык | 고자원 언어 |
| low-resource language | малоресурсный язык | 저자원 언어 |
| writing system | письменность | 문자 체계 |
| tokenizer fertility | фертильность токенизатора | 토크나이저 생산성 |
| morphologically rich language | морфологически богатый язык | 형태론적으로 풍부한 언어 |
| vocabulary allocation | распределение словаря | 어휘 할당 |
| temperature sampling | температурное сэмплирование | 온도 샘플링 |
| curse of multilinguality | проклятие многоязычности | 다국어성의 저주 |
| code-switching | переключение кодов | 코드 전환 |
| transliteration | транслитерация | 음역 |
| cultural bias | культурная предвзятость | 문화적 편향 |

---
[← Previous: Mixture of Experts](29-mixture-of-experts.md) · [Contents](README.md) · [Next: Fine-tuning and Transfer Learning →](31-fine-tuning.md)
