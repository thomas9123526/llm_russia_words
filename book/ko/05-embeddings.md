# 제5장. 임베딩: 벡터로서의 단어(Word2Vec, GloVe)

> **제1부 — 텍스트와 표현** · 수준: 초급
>
> **이 장에서 배울 내용:** 단어를 밀집 벡터로 표현해야 하는 이유 · 이웃 단어로부터 단어의 의미를 학습하는 방법(Word2Vec, GloVe) · 임베딩이 잘 포착하는 것과 실패하는 지점

## 모델은 두 단어가 비슷하다는 것을 어떻게 알 수 있을까?

제4장 끝에서 우리는 세기의 근본적인 약점을 발견했습니다. N-그램 모델에게 *cat*과 *kitten*은 *cat*과 *inflation*만큼이나 관계가 없습니다. 제2장의 원-핫 벡터도 같은 문제를 가집니다. 두 벡터 $\mathbf{a}$와 $\mathbf{b}$의 **내적**(dot product)은 $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$이며, 서로 다른 두 원-핫 벡터의 내적은 항상 0입니다. 모든 단어가 다른 모든 단어로부터 똑같이 떨어져 있습니다.

우리는 더 나은 것을 원합니다. 각 단어가 공간 속의 점이고, 비슷한 의미를 가진 단어들이 서로 가까이 있는 표현입니다. 그러면 모델이 *cat*에 대해 배운 것이 일부 *kitten*으로 옮겨 갑니다. 이런 표현을 **단어 임베딩**(word embedding)이라고 합니다. 각 단어를 길이 50,000의 원-핫 벡터 대신 길이 100이나 300 정도의 짧은 실수 벡터로 대응시키는("심는", embed) 것입니다. 원-핫 벡터와 달리 이 벡터들은 **밀집**(dense) 벡터, 즉 성분 대부분이 0이 아닌 벡터입니다.

그런데 이 숫자들은 어디서 올까요? 누구도 이를 손으로 쓸 수 없습니다. 이 장에서는 날것의 텍스트로부터 이를 학습하는 방법을 보여 줍니다.

## 단어는 그 단어와 어울리는 친구로 알 수 있다

작은 실험을 해 봅시다. 여러분은 아마 *tezgüino*라는 단어를 모를 것입니다. 이제 다음을 읽어 보십시오.

- *tezgüino 한 병이 탁자 위에 있다.*
- *모두가 tezgüino를 좋아한다.*
- *tezgüino를 마시면 취한다.*
- *우리는 옥수수로 tezgüino를 만든다.*

네 문장을 읽고 나면 *tezgüino*가 맥주와 비슷한 일종의 술이라고 짐작할 수 있습니다. 여러분은 오직 주변 단어들로부터 이것을 알아냈습니다. 이 아이디어가 **분포 가설**(distributional hypothesis)입니다. 비슷한 문맥에 나타나는 단어는 비슷한 의미를 가지는 경향이 있다는 것입니다. 이는 1950년대 언어학자 젤리그 해리스(Zellig Harris)가 정식화했고, 존 퍼스(John Firth)는 이를 유명한 문장으로 요약했습니다. *"You shall know a word by the company it keeps"*(단어는 그 단어와 어울리는 친구로 알 수 있다).

이 가설은 우리에게 방법을 알려 줍니다. 각 단어를 그 단어가 나타나는 문맥으로 기술하면, 의미가 비슷한 단어들은 비슷한 기술을 얻게 됩니다. 가장 간단한 방법은 세는 것입니다. **문맥 창**(context window), 예를 들어 좌우로 한두 단어를 정하고, 크기 $V \times V$의 **동시 출현 행렬**(co-occurrence matrix) $X$를 만듭니다. 여기서 $X_{ij}$는 단어 $i$ 주변의 창 안에 단어 $j$가 나타난 횟수입니다. $X$의 각 행이 한 단어에 대한 기술입니다. *tea*와 *coffee*가 둘 다 *drink*와 *like* 뒤에 나타난다면 두 행은 비슷해질 것입니다.

동시 출현 행은 길고(길이 $V$) 희소하며, *the* 같은 빈번한 단어가 지배합니다. 그래서 고전적 방법들은 빈도의 가중치를 다시 매깁니다. 예를 들어 **점별 상호 정보량**(pointwise mutual information, PMI)을 쓰는데, 이는 제4장의 TF-IDF처럼 우연보다 더 자주 함께 나타나는 쌍에 보상을 줍니다. 그다음 특잇값 분해(SVD) 같은 **행렬 분해**(matrix factorization) 방법으로 행렬을 압축합니다. 그 결과 각 단어에 대해 짧은 밀집 벡터가 얻어집니다. 잠재 의미 분석이라고 알려진 이 접근법은 놀라울 만큼 잘 작동했습니다. 다음에 공부할 Word2Vec과 GloVe는 같은 일을 더 효율적이고 정확하게 하는 방법이라고 볼 수 있습니다.

## 유사도 측정: 코사인

단어가 벡터가 되었으니, 두 벡터가 얼마나 비슷한지 말할 방법이 필요합니다. 표준적인 선택은 두 벡터 사이 각도의 코사인인 **코사인 유사도**(cosine similarity)입니다.

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert},$$

여기서 $\lVert \mathbf{a} \rVert = \sqrt{\sum_i a_i^2}$는 $\mathbf{a}$의 길이입니다. 값은 두 벡터가 같은 방향이면 1, 수직이면 0, 반대 방향이면 $-1$입니다. 작은 예를 봅시다. $\mathbf{a} = (1, 2)$, $\mathbf{b} = (2, 3)$이면 $\mathbf{a} \cdot \mathbf{b} = 8$, $\lVert \mathbf{a} \rVert = \sqrt{5}$, $\lVert \mathbf{b} \rVert = \sqrt{13}$이므로 $\cos = 8 / \sqrt{65} \approx 0.99$입니다. 코사인은 벡터의 길이를 무시하고 방향만 봅니다. 단어 벡터의 길이는 의미보다 단어 빈도를 반영하는 경우가 많으므로 이는 유용합니다.

어휘 집합의 모든 임베딩은 크기 $V \times d$인 **임베딩 행렬**(embedding matrix) $E$에 함께 저장되며, 여기서 $d$는 임베딩 차원입니다. $E$의 $i$번째 행이 단어 $i$의 벡터입니다. 토큰의 임베딩을 얻는 일은 그 인덱스(제2장)로 행 하나를 찾아보는 것뿐입니다. 수학적으로는 토큰의 원-핫 벡터에 $E$를 곱하는 것과 같지만, 실제 곱셈은 필요 없습니다. 이 조회 테이블이 바로 모든 현대 언어 모델의 첫 번째 층이며, PyTorch에서는 `nn.Embedding`이라고 부릅니다(제20장).

## Word2Vec

2013년 구글의 미콜로프(Mikolov)와 동료들은 세기 대신 *예측*으로 임베딩을 학습하는 빠른 방법들의 모음인 **Word2Vec**을 소개했습니다. 두 가지 변형이 있습니다.

- **스킵그램**(skip-gram): 단어가 주어지면 그 문맥 창 안의 단어들을 예측합니다.
- **CBOW**(continuous bag of words, 연속 단어 주머니): 문맥 단어들이 주어지면 가운데 단어를 예측합니다.

스킵그램을 살펴봅시다. 각 단어 $w$는 두 개의 벡터를 가집니다. "목표" 벡터 $\mathbf{v}_w$($w$가 중심 단어일 때 사용)와 "문맥" 벡터 $\mathbf{u}_w$($w$가 문맥에 나타날 때 사용)입니다. 모델은 **소프트맥스**(softmax) 함수로 두 벡터의 내적을 확률로 바꿉니다.

$$P(c \mid w) = \frac{\exp(\mathbf{u}_c \cdot \mathbf{v}_w)}{\sum_{c'=1}^{V} \exp(\mathbf{u}_{c'} \cdot \mathbf{v}_w)}.$$

분자는 $\mathbf{u}_c$와 $\mathbf{v}_w$가 비슷한 방향일 때 커지고, 어휘 집합 전체에 대한 합인 분모는 확률의 합이 1이 되게 합니다. 학습은 실제 문맥 단어가 높은 확률을 얻도록 모든 벡터를 조정합니다.

문제는 분모입니다. 학습 쌍마다 분모를 계산하려면 수만 개 단어에 대한 합이 필요합니다. Word2Vec은 **네거티브 샘플링**(negative sampling)으로 이를 피합니다. 전체 소프트맥스 대신 모델은 더 단순한 예/아니요 과제를 풉니다. 쌍 $(w, c)$가 텍스트에서 온 실제 쌍인가, 아니면 무작위 쌍인가? 실제 쌍마다 무작위 "네거티브" 단어 $k$개 $n_1, \dots, n_k$를 뽑고 다음을 최소화합니다.

$$L = -\log \sigma(\mathbf{u}_c \cdot \mathbf{v}_w) - \sum_{j=1}^{k} \log \sigma(-\mathbf{u}_{n_j} \cdot \mathbf{v}_w),$$

여기서 $\sigma(x) = 1 / (1 + e^{-x})$는 임의의 수를 구간 $(0, 1)$로 압축하는 **시그모이드**(sigmoid) 함수입니다. 첫째 항은 실제 쌍의 점수를 올리고, 둘째 항은 무작위 쌍의 점수를 내립니다. 원래 방법에서는 네거티브 단어를 빈도의 $3/4$제곱에 비례하는 확률로 뽑으며, $k$는 대략 2에서 20 사이입니다.

다음은 장난감 말뭉치로 학습하는, NumPy로 작성한 네거티브 샘플링 스킵그램의 완전한 구현입니다. 여기서는 경사 하강법을 쓰는데, 이는 제8장에서 제대로 공부합니다. 지금은 각 갱신이 $L$을 줄이는 방향으로 벡터를 조금씩 움직인다는 것만 알면 충분합니다. 각 점수 $s = \mathbf{u} \cdot \mathbf{v}$에 대한 $L$의 도함수는 간단히 $\sigma(s) - y$이며, 여기서 실제 쌍이면 $y = 1$, 네거티브이면 $y = 0$입니다.

```python
import numpy as np

corpus = ["i like tea", "i like coffee", "i drink tea", "i drink coffee",
          "dogs chase cats", "cats chase mice", "dogs chase mice"]
words = sorted({w for s in corpus for w in s.split()})
idx = {w: i for i, w in enumerate(words)}
rng = np.random.default_rng(0)
V, d, k, lr = len(words), 5, 3, 0.05   # 어휘 크기, 차원, 네거티브 수, 학습 보폭
W_in = rng.normal(0, 0.1, (V, d))      # 목표 단어 벡터(우리가 얻을 임베딩)
W_out = rng.normal(0, 0.1, (V, d))     # 문맥 단어 벡터
sigmoid = lambda x: 1 / (1 + np.exp(-x))

for epoch in range(300):
    for t in (s.split() for s in corpus):
        for i, w in enumerate(t):
            for j in range(max(0, i - 1), min(len(t), i + 2)):   # 크기 1의 창
                if j == i:
                    continue
                # 실제 문맥 단어 1개 + 무작위 "네거티브" 단어 k개(여기서는 균등 추출)
                ids = np.array([idx[t[j]]] + list(rng.integers(0, V, k)))
                labels = np.array([1.0] + [0.0] * k)
                v, u = W_in[idx[w]], W_out[ids]
                g = sigmoid(u @ v) - labels        # 점수에 대한 손실의 기울기
                W_in[idx[w]] -= lr * g @ u
                np.add.at(W_out, ids, -lr * np.outer(g, v))

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
for a, b in [("tea", "coffee"), ("like", "drink"), ("tea", "dogs")]:
    print(a, b, round(float(cosine(W_in[idx[a]], W_in[idx[b]])), 2))
```

프로그램은 `tea coffee 1.0`, `like drink 1.0`, `tea dogs 0.49`를 출력합니다. *tea*와 *coffee*는 한 번도 함께 나타나지 않지만 같은 문맥에 나타나므로 벡터가 거의 같아집니다. *like*와 *drink*도 마찬가지입니다. 관계없는 쌍 *tea*–*dogs*는 훨씬 덜 비슷합니다. (정확한 숫자는 난수 시드에 따라 달라지며, 실제 말뭉치에서는 관계없는 단어의 코사인이 보통 0에 가깝습니다.) 같은 단어가 뽑힌 ids 안에 두 번 나올 수 있으므로 `np.add.at`을 사용합니다.

## GloVe

2014년 페닝턴(Pennington), 소처(Socher), 매닝(Manning)이 제안한 **GloVe**(Global Vectors)는 두 접근법을 연결합니다. 먼저 세기 기반 방법처럼 전역 동시 출현 행렬 $X$를 만들고, 그다음 내적이 빈도의 로그를 예측하도록 벡터를 학습합니다.

$$J = \sum_{i,j} f(X_{ij}) \left( \mathbf{w}_i \cdot \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2.$$

여기서 $\mathbf{w}_i$는 단어 $i$의 벡터, $\tilde{\mathbf{w}}_j$는 단어 $j$의 문맥 벡터, $b_i$와 $\tilde{b}_j$는 스칼라 편향이며, 합은 $X_{ij} > 0$인 쌍에 대해 취합니다. 가중 함수 $f$는 (빈도에 잡음이 많은) 드문 쌍에는 작은 가중치를 주고, 매우 빈번한 쌍의 가중치에는 상한을 둡니다. 실제로 Word2Vec과 GloVe는 비슷한 품질의 임베딩을 줍니다. 나중에 네거티브 샘플링 스킵그램이 사실상 이동된 PMI 행렬을 암묵적으로 분해하고 있다는 것이 밝혀졌으므로, 두 계열은 겉보기보다 가깝습니다.

## 임베딩이 포착하는 것과 놓치는 것

대규모 말뭉치로 학습한 단어 임베딩은 놀라운 규칙성을 보입니다. 가장 유명한 것은 **벡터 산술**(vector arithmetic)을 이용한 **단어 유추**(word analogy)입니다.

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}.$$

차이 $\mathbf{v}_{\text{woman}} - \mathbf{v}_{\text{man}}$는 "성별 방향"처럼 작동합니다. 나라–수도(*Франция → Париж, Россия → Москва*)나 문법(*идти → шёл*, "가다 → 갔다")에 대해서도 비슷한 방향이 나타납니다. 이런 유추는 완벽하지 않고, 계산된 벡터에 가장 가까운 단어를 찾는 방식으로 얻어지지만, 공간에 의미 있는 구조가 있음을 보여 줍니다.

고전적인 단어 임베딩에는 분명한 한계도 있습니다.

1. **미지 단어와 드문 단어.** 각 단어 형태가 자신만의 벡터를 가지므로, 학습 중에 보지 못한 형태에는 벡터가 없습니다. 이는 한 단어가 많은 형태를 가지는 러시아어에 불리합니다. **fastText**(Bojanowski 외, 2017)는 단어를 문자 N-그램 벡터들의 합으로 표현합니다. 그래서 *котами*는 *кот*, *котом*과 조각을 공유하고, 보지 못한 형태도 벡터를 얻습니다. 이는 제3장의 서브워드 토큰화와 같은 아이디어입니다.
2. **단어당 벡터 하나.** 이런 임베딩은 **정적**(static)입니다. 단어는 어떤 문장에서든 같은 벡터를 가집니다. 그러나 러시아어 *ключ*는 열쇠일 수도 있고 샘일 수도 있으며, 한국어 *배*는 과일, 선박, 신체 부위일 수 있습니다. 정적 벡터는 모든 의미를 한데 섞어 버립니다.
3. **편향.** 임베딩은 텍스트에 있는 모든 규칙성을 학습하며, 여기에는 직업과 성별의 연관 같은 사회적 고정관념도 포함됩니다. 이 문제는 제40장에서 다시 다룹니다.

두 번째 한계가 현대 언어 모델의 핵심 아이디어로 이어집니다. 트랜스포머는 단어당 고정된 벡터 하나 대신, 정적 임베딩($E$의 한 행)에서 출발하여 문장 전체에 의존하는 각 토큰의 **문맥적 표현**(contextual representation)을 계산합니다. 그래서 *«ключ от двери»*(문 열쇠)의 *ключ*와 *«горный ключ»*(산속 샘)의 *ключ*는 서로 다른 벡터를 얻습니다. 이런 신경망을 어떻게 만들고 학습시키는지 이해하려면, 먼저 제2부의 도구들, 즉 벡터와 행렬, 층, 손실 함수, 경사 하강법이 필요합니다.

## 요약

- 원-핫 벡터는 모든 단어를 똑같이 다르게 만듭니다. 단어 임베딩은 각 단어를 비슷한 단어끼리 가까운 짧은 밀집 벡터로 대응시킵니다.
- 분포 가설(비슷한 문맥의 단어는 비슷한 의미를 가진다) 덕분에, 동시 출현을 세어 행렬을 분해하거나 예측을 통해 날것의 텍스트로부터 임베딩을 학습할 수 있습니다.
- 코사인 유사도는 두 벡터 사이의 각도로 가까움을 측정합니다. 모든 임베딩은 임베딩 행렬을 이루며, 조회는 행 선택입니다.
- Word2Vec(스킵그램, CBOW)은 문맥 단어를 예측하여 벡터를 학습하며, 네거티브 샘플링으로 효율을 높입니다. GloVe는 내적을 전역 동시 출현 빈도의 로그에 맞춥니다.
- 임베딩은 유추를 포착하지만, 정적 단어 벡터는 보지 못한 형태를 다루지 못하고(fastText가 도움이 됨), 다의어의 의미를 섞으며, 편향을 흡수합니다. 트랜스포머의 문맥적 표현은 두 번째 문제를 해결합니다.

## 연습 문제

1. 분포 가설을 자신의 말로 설명하고, 문맥으로부터 의미를 짐작할 수 있는 단어의 예를 러시아어나 한국어로 직접 드십시오.
2. $\mathbf{a} = (1, 0, 2)$와 $\mathbf{b} = (2, 1, 1)$의 코사인 유사도를 손으로 계산한 다음 NumPy로 확인하십시오.
3. 스킵그램 코드에서 차원 `d`를 2로, 시드를 1, 2, 3으로 바꿔 보십시오. 유사도는 얼마나 안정적입니까? 그다음 문장 *"i like juice"*와 *"i drink juice"*를 추가하십시오. *juice*에 가장 가까워지는 단어는 무엇입니까?
4. 정적 임베딩은 왜 *ключ*의 두 의미를 구별할 수 없습니까? 모델이 이를 구별하려면 어떤 정보가 필요한지 두세 문장으로 설명하십시오.

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 단어 임베딩 | word embedding | векторное представление слова | véktornoye predstavléniye slóva |
| 밀집 벡터 | dense vector | плотный вектор | plótnyy véktor |
| 분포 가설 | distributional hypothesis | дистрибутивная гипотеза | distributívnaya gipóteza |
| 문맥 창 | context window | контекстное окно | kontékstnoye oknó |
| 동시 출현 행렬 | co-occurrence matrix | матрица совместной встречаемости | mátritsa sovméstnoy vstrecháyemosti |
| 점별 상호 정보량(PMI) | pointwise mutual information | поточечная взаимная информация | potóchechnaya vzaímnaya informátsiya |
| 내적 | dot product | скалярное произведение | skalyárnoye proizvedéniye |
| 코사인 유사도 | cosine similarity | косинусное сходство | kósinusnoye skhódstvo |
| 임베딩 행렬 | embedding matrix | матрица векторных представлений | mátritsa véktornykh predstavléniy |
| 네거티브 샘플링 | negative sampling | отрицательное сэмплирование | otritsátel'noye semplírovaniye |
| 소프트맥스 | softmax | функция softmax | fúnktsiya softmáks |
| 시그모이드 | sigmoid | сигмоида | sigmóida |
| 단어 유추 | word analogy | аналогия слов | analógiya slov |
| 벡터 산술 | vector arithmetic | векторная арифметика | véktornaya arifmétika |
| 문맥적 표현 | contextual representation | контекстное представление | kontékstnoye predstavléniye |

---
[← 이전: 단어 세기: 단어 주머니, N-그램, 퍼플렉서티](04-counting-words.md) · [목차](README.md) · [다음: 필요한 수학: 벡터, 행렬, 확률 →](06-math-you-need.md)
