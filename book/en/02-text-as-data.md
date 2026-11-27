# Chapter 2. Text as Data: Characters, Words and Encodings

> **Part 1 — Text and Representation** · Level: Beginner
>
> **In this chapter you will learn:** how a computer stores characters as numbers and bytes · why the same text can have several binary forms · why "a word" is harder to define than it seems

## How long is a word?

Take three words that mean the same thing: English *cat*, Russian *кот* and Korean *고양이*. Each has three characters. But if you save each word in a file, the files have different sizes: 3 bytes, 6 bytes and 9 bytes. Why?

In Chapter 1 we said that a language model works with a sequence of tokens. Before we can split text into tokens, we must understand what text *is* inside a computer. A computer stores only numbers. Every letter, digit, space and emoji must therefore be turned into numbers, and those numbers must be written as bytes. This chapter explains how that is done and why the details matter for language models, especially for languages like Russian and Korean.

## Characters and code points

The first step is to give every character a number. The international standard for this is **Unicode**. Unicode assigns each character a unique integer called a **code point**, written in hexadecimal with the prefix `U+`. For example:

- Latin *c* is U+0063 (decimal 99);
- Cyrillic *к* is U+043A (decimal 1082);
- Hangul syllable *고* is U+ACE0 (decimal 44256).

Code points range from U+0000 to U+10FFFF, which leaves room for more than a million characters. They cover the alphabets of the world, mathematical symbols, emoji and much more. A text, at this level, is simply a sequence of code points. In Python, `ord("к")` gives the code point of a character and `chr(1082)` goes back.

Korean is an interesting case. Modern Hangul has 19 initial consonants, 21 vowels and 28 possible final parts (27 final consonants plus "no final"). Unicode contains every possible syllable as a separate character, $19 \cdot 21 \cdot 28 = 11{,}172$ in total, in the block starting at U+AC00. The code point of a syllable is computed by a simple formula:

$$\text{code} = \text{0xAC00} + (21 \cdot i + m) \cdot 28 + f,$$

where $i$ is the index of the initial consonant (0–18), $m$ is the index of the vowel (0–20), and $f$ is the index of the final consonant (0–27, where 0 means "no final"). For *한* we have ㅎ with $i = 18$, ㅏ with $m = 0$ and ㄴ with $f = 4$, so the code is $44032 + 378 \cdot 28 + 4 = 54620$, which is U+D55C.

## From code points to bytes: encodings

A code point is an abstract number. To store it in memory or in a file, we need an **encoding**: a rule that turns code points into a sequence of **bytes**. A byte is 8 bits and can hold a value from 0 to 255.

The oldest widely used encoding is **ASCII**. It defines only 128 characters (English letters, digits, punctuation and control codes), each stored in one byte. ASCII has no Cyrillic and no Hangul. For decades, different countries used their own incompatible encodings (for Russian, for example, KOI8-R and Windows-1251; for Korean, EUC-KR). A file written in one encoding turned into garbage when read in another. Russians even have a word for this garbage: *кракозябры*.

Today the dominant encoding is **UTF-8**. It uses a variable number of bytes per code point:

| Code point range | Bytes | Examples |
|---|---|---|
| U+0000 – U+007F | 1 | English letters, digits (same as ASCII) |
| U+0080 – U+07FF | 2 | Cyrillic, Greek, Arabic |
| U+0800 – U+FFFF | 3 | Hangul, Chinese characters |
| U+10000 – U+10FFFF | 4 | emoji, rare scripts |

Now we can answer our opening question: *cat* takes $3 \cdot 1 = 3$ bytes, *кот* takes $3 \cdot 2 = 6$ bytes, and *고양이* takes $3 \cdot 3 = 9$ bytes. UTF-8 has two great advantages. Any ASCII text is already valid UTF-8, and the bytes of a multi-byte character can never be confused with an ASCII character, so a program can always find where characters begin.

The following code shows code points and UTF-8 bytes for our three words.

```python
import unicodedata

for word in ["cat", "кот", "고양이"]:
    codes = [f"U+{ord(ch):04X}" for ch in word]   # Unicode code points
    data = word.encode("utf-8")                    # UTF-8 bytes
    print(word, len(word), codes, len(data), list(data))

# One visible character, two possible representations
for s in ["ё", "한"]:
    nfd = unicodedata.normalize("NFD", s)          # decomposed form
    print(s, len(s), len(nfd), [f"U+{ord(c):04X}" for c in nfd])
    print(nfd == s, unicodedata.normalize("NFC", nfd) == s)
```

The first loop prints, for example, `кот 3 ['U+043A', 'U+043E', 'U+0442'] 6 [208, 186, 208, 190, 209, 130]`: three characters, six bytes. The second loop is the subject of the next section.

Why should a language model care about bytes? Some modern tokenizers start not from characters but from the 256 possible byte values (we will see this "byte-level" approach in Chapter 3). For them, a Korean syllable is three units before any merging, and a Russian letter is two. This is one reason why the same sentence can cost a different number of tokens in different languages.

## Normalization: one text, several forms

Unicode allows some characters to be written in more than one way. The Russian letter *ё* can be one code point (U+0451) or two: the letter *е* (U+0435) followed by a combining diaeresis (U+0308). Both look identical on the screen. In the same way, the Korean syllable *한* can be stored as one precomposed code point (U+D55C) or as three separate jamo: ㅎ, ㅏ, ㄴ (U+1112, U+1161, U+11AB). The second loop of our program shows exactly this: the decomposed form of *ё* has length 2, the decomposed form of *한* has length 3, and they are *not* equal to the originals as strings.

For a computer, two strings that differ in even one code point are different. If the training data contains both forms, a model sees two "different" words where a human sees one. The solution is **text normalization**: converting text into one standard form before any further processing. Unicode defines standard normalization forms, of which the most common are:

- **NFC** (composed): combine characters into precomposed code points where possible;
- **NFD** (decomposed): split characters into base characters and combining marks.

Our program confirms that NFC turns the decomposed forms back into the originals (`True` in the last column). Most NLP pipelines apply NFC or the related **NFKC** form, which also maps "compatibility" characters, such as full-width Latin letters, to their ordinary forms.

Normalization can go further, and each step is a design choice. **Lowercasing** makes *Кот* and *кот* the same token, but it also loses information (*Apple* the company vs. *apple* the fruit). In Russian, many texts write *е* instead of *ё* (*еще* instead of *ещё*), so some systems replace *ё* with *е* everywhere. Modern large language models usually apply only light normalization and keep case, punctuation and spacing, because they want to generate text exactly as people write it.

## What is a word?

So far we have discussed characters. But people think in words, and early NLP systems treated words as the basic units. What exactly is a word?

In English, a first answer is "a sequence of letters between spaces". This already has problems: is *don't* one word or two? Is *New York* one unit? What about *state-of-the-art*? The process of splitting text into units is called **tokenization**, and even for English the rules are not obvious.

For other languages the problems become serious:

- **Russian** has rich **morphology** (the way words change form). One noun has up to about a dozen forms: *кот, кота, коту, котом, коте, коты, котов, котам…* A verb has many more. If every form is a separate word, the vocabulary becomes huge and each form is seen less often in the data.
- **Korean** is agglutinative: grammatical particles and endings attach to the word. *고양이가, 고양이를, 고양이는, 고양이에게* all contain *고양이*, but a space-based splitter treats them as four unrelated words. Korean spacing rules are also complex, and people often break them in informal writing.
- **Chinese** and **Japanese** do not use spaces between words at all, so "split on spaces" does not work.

Linguists call the smallest meaningful part of a word a **morpheme**: in *котами*, the stem *кот-* carries the meaning "cat" and the ending *-ами* means "instrumental plural". Ideally, a model would know that all forms of a word share a meaning. These difficulties are the main motivation for **subword tokenization**, the topic of Chapter 3: instead of choosing between characters (too small) and words (too many), we learn units in between.

## From symbols to numbers

Whatever units we choose — characters, bytes, words or subwords — the last step is the same. We fix a vocabulary, a list of all units, and give each unit an integer **index** (also called a **token ID**). For example, with the vocabulary `["кот", "собака", "чай"]`, the word *кот* gets index 0, *собака* index 1 and *чай* index 2. A text becomes a list of integers, and this list is what a model actually receives.

An index is just a name; the number 2 is not "bigger" than 0 in any meaningful sense. To give a neural network an input without such false order, we can use a **one-hot vector**: a vector of length $V$ (the vocabulary size) with a 1 at the position of the token and 0 everywhere else. With our tiny vocabulary,

$$\text{кот} = (1, 0, 0), \quad \text{собака} = (0, 1, 0), \quad \text{чай} = (0, 0, 1).$$

One-hot vectors are simple but wasteful: with a vocabulary of 50,000 tokens, each vector has 49,999 zeros. They also say nothing about meaning: the vectors for *кот* and *собака* are exactly as different as those for *кот* and *чай*. In Chapter 5 we will replace them with short dense vectors, called embeddings, where similar words get similar vectors.

## Summary

- Unicode assigns every character a code point; a text is a sequence of code points.
- An encoding such as UTF-8 turns code points into bytes; in UTF-8 an English letter takes 1 byte, a Cyrillic letter 2 bytes and a Hangul syllable 3 bytes.
- The same visible text can have different code-point sequences (for example, precomposed vs. decomposed *ё* or *한*); normalization (NFC, NFKC) makes them identical.
- "Word" is not a simple notion: Russian morphology, Korean particles and languages without spaces all make word-based processing difficult.
- Finally, every unit gets an integer index; one-hot vectors represent indices without false order, but they are large and carry no meaning.

## Exercises

1. Explain in your own words the difference between a character, a code point and a byte.
2. Using the formula for Hangul syllables, compute the code point of *각* (ㄱ: $i = 0$; ㅏ: $m = 0$; ㄱ as final: $f = 1$). Check your answer with `ord("각")` in Python.
3. How many bytes does the UTF-8 encoding of the string *"Привет, 세계!"* take? First compute it by hand, then check with `len(s.encode("utf-8"))`.
4. Write down ten forms of the Russian word *книга* and five forms of the Korean word *책* with particles. What problems would a word-level vocabulary have with them?

## Key terms

| English | Русский | 한국어 |
|---|---|---|
| character | символ | 문자 |
| Unicode | Юникод | 유니코드 |
| code point | кодовая точка | 코드 포인트 |
| character encoding | кодировка символов | 문자 인코딩 |
| byte | байт | 바이트 |
| UTF-8 | UTF-8 | UTF-8 |
| text normalization | нормализация текста | 텍스트 정규화 |
| lowercasing | приведение к нижнему регистру | 소문자화 |
| tokenization | токенизация | 토큰화 |
| morphology | морфология | 형태론 |
| morpheme | морфема | 형태소 |
| subword | подслово | 서브워드 |
| token ID (index) | индекс токена | 토큰 ID(인덱스) |
| one-hot vector | унитарный вектор (one-hot) | 원-핫 벡터 |
| vocabulary size | размер словаря | 어휘 집합 크기 |

---
[← Previous: What Is a Language Model?](01-what-is-a-language-model.md) · [Contents](README.md) · [Next: Tokenization and Subwords (BPE, WordPiece, SentencePiece) →](03-tokenization.md)
