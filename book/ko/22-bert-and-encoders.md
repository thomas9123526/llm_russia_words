# 제22장. BERT와 인코더 모델

> **제5부 — 실전 언어 모델** · 수준: 중급
>
> **이 장에서 배울 내용:** BERT의 입력과 아키텍처가 구성되는 방식 · 사전 학습된 하나의 인코더를 분류, 태깅, 질의응답에 맞게 미세 조정하는 방법 · BERT 계열의 주요 모델과 오늘날 인코더가 쓰이는 곳

## 읽기 과제

일상적인 세 가지 문제를 생각해 봅시다.

1. *"배터리가 두 시간 만에 닳는다"*라는 리뷰는 긍정일까요, 부정일까요?
2. *"이반 페트로프는 모스크바의 얀덱스에서 일한다"*에서 사람, 조직, 장소의 이름은 어떤 단어일까요?
3. 볼가 강에 관한 문단과 *"볼가 강은 어디로 흘러 들어가는가?"*라는 질문이 주어졌을 때, 문단의 어느 부분이 답일까요?

이 과제들은 어느 것도 새 텍스트를 *쓰는* 일을 요구하지 않습니다. 대신 잘 *읽어야* 합니다. 각 단어를 문장 전체의 문맥, 즉 앞에 오는 것과 뒤에 오는 것 모두 속에서 이해해야 합니다. 제21장에서 마스크 언어 모델링이 바로 이 능력을 훈련한다는 것을 보았습니다. 이 장에서는 이를 유명하게 만든 모델인 **BERT**(Bidirectional Encoder Representations from Transformers, Devlin et al., 2019)와 그 뒤를 이은 **인코더 전용 모델**(encoder-only model) 계열을 공부합니다.

## 아키텍처와 입력 형식

BERT는 인과적 마스크 없이 트랜스포머 인코더 블록(제18장)을 쌓은 것일 뿐입니다. 원 논문은 두 가지 크기를 제시했습니다.

| 모델 | 층 수 | 은닉 차원 $d$ | 헤드 수 | 파라미터 수 |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 약 1억 1천만 |
| BERT-large | 24 | 1024 | 16 | 약 3억 4천만 |

흥미로운 부분은 입력입니다. 텍스트는 약 3만 개의 서브워드 어휘를 가진 WordPiece **토크나이저**(tokenizer, 제3장)로 분할됩니다. 그다음 두 개의 **특수 토큰**(special token)이 추가됩니다.

- `[CLS]`("classification", 분류)는 맨 앞에 붙습니다. 그 출력 벡터는 입력 전체의 요약으로 사용됩니다.
- `[SEP]`("separator", 구분자)는 각 세그먼트의 끝에 붙습니다. 많은 과제가 질문과 문단, 또는 두 문장처럼 *두 개의* 텍스트를 입력으로 받으며, `[SEP]`가 그 경계를 표시합니다.

문장 쌍의 입력은 다음과 같습니다.

```text
[CLS] the cat sat [SEP] it was tired [SEP]
  A    A   A   A    A   B   B    B     B
```

둘째 줄은 **세그먼트 임베딩**(segment embedding)을 나타냅니다. 첫 번째 텍스트의 모든 토큰에는 학습되는 벡터 $s_A$가, 두 번째 텍스트의 모든 토큰에는 $s_B$가 주어집니다. 위치 $i$의 토큰에 대한 입력 벡터는 세 임베딩의 합입니다.

$$h_i^{(0)} = e_{\text{tok}}(x_i) + e_{\text{pos}}(i) + e_{\text{seg}}(i)$$

여기서 $e_{\text{tok}}$는 토큰 임베딩(제5장), $e_{\text{pos}}$는 학습되는 위치 임베딩(제17장), $e_{\text{seg}}$는 $s_A$ 또는 $s_B$입니다. $L$개의 인코더 층을 지나면 각 토큰마다 하나의 문맥 벡터 $h_i^{(L)}$을 얻습니다. Word2Vec의 정적 벡터와 달리 이 벡터들은 문맥에 따라 달라집니다. *"river bank"*(강둑)의 *"bank"*와 *"bank account"*(은행 계좌)의 *"bank"*는 서로 다른 표현을 받습니다.

배치 안의 문장들은 길이가 다르므로 짧은 문장은 `[PAD]` 토큰으로 채웁니다. 이 **패딩**(padding)은 결과에 영향을 주면 안 되므로, **패딩 마스크**(padding mask)가 어텐션에게 그 위치들을 무시하라고 알려 줍니다.

## 사전 학습: MLM과 다음 문장 예측

BERT는 두 가지 목표로 동시에 사전 학습되었습니다.

- **마스크 언어 모델링**: 제21장과 똑같이 위치의 15%에 80/10/10 규칙을 적용합니다.
- **다음 문장 예측**(next sentence prediction, NSP). 모델은 세그먼트 A와 B의 쌍을 받습니다. 예제의 50%에서는 말뭉치에서 B가 실제로 A 뒤에 오고, 나머지 50%에서는 B가 다른 문서의 무작위 문장입니다. $h_{\text{[CLS]}}$ 위의 작은 분류기가 어느 경우인지 예측합니다. 질의응답과 추론 과제에 유용한 문장 간 관계를 모델에게 가르치려는 의도였습니다.

이후 연구는 NSP에 의문을 제기했습니다. **RoBERTa**(Liu et al., 2019)는 NSP를 없애고, 훨씬 많은 데이터로 더 오래 학습했으며, 더 큰 배치를 쓰고, 문장을 다시 볼 때마다 새로운 마스크 위치를 골랐습니다(*동적 마스킹*). 결과는 원래 BERT보다 분명히 좋았습니다. 이것은 분야 전체에 중요한 교훈이었습니다. 많은 "아키텍처" 개선이 단순히 잘, 그리고 오래 학습하는 효과보다 작은 것으로 드러난다는 것입니다. 이 주제는 스케일링 법칙을 다루는 제27장에서 다시 다룹니다.

## 미세 조정: 하나의 인코더, 여러 헤드

BERT의 힘은 **전이 학습**(transfer learning)에서 나옵니다. 레이블 없는 텍스트로 한 번 사전 학습한 다음, 특정 과제의 레이블된 데이터셋으로 네트워크 전체와 작은 새 층을 함께 미세 조정합니다. 이 과제별 작은 층을 **헤드**(head)라고 합니다. 인코더가 이미 언어를 "알고" 있으므로 **미세 조정**(fine-tuning)에는 보통 수천 개의 레이블된 예제와 몇 에포크면 충분합니다.

이 장 처음의 세 가지 읽기 과제는 세 가지 표준 헤드에 대응합니다.

**시퀀스 분류**(감성, 주제, 스팸). `[CLS]`의 출력 벡터에 선형 층과 소프트맥스를 적용합니다.

$$p(y \mid x) = \operatorname{softmax}\left(W h_{\text{[CLS]}}^{(L)} + b\right)$$

여기서 $W \in \mathbb{R}^{C \times d}$와 $b \in \mathbb{R}^{C}$는 새 파라미터이고 $C$는 클래스 수입니다. 이 **분류 헤드**(classification head)는 사전 학습 동안 `[CLS]`가 모든 층에서 모든 토큰에 어텐션했다는 사실에 기대고 있습니다. 그 덕분에 미세 조정 후 그 벡터는 과제에 필요한 정보를 정확히 모으도록 학습합니다.

**토큰 분류**(**개체명 인식**(named entity recognition), 품사 태깅). 같은 선형 층을 *모든* 출력 벡터 $h_i^{(L)}$에 적용하여 토큰마다 하나의 레이블, 예를 들어 `B-PER`, `I-PER`, `B-ORG`, `O`를 얻습니다. 한 단어가 여러 서브워드로 나뉠 수 있으므로 보통 각 단어의 첫 서브워드만 레이블을 받습니다.

**추출형 질의응답**(question answering). 질문이 세그먼트 A, 문단이 세그먼트 B입니다. 답은 문단의 한 구간이므로 모델은 그 시작과 끝을 예측해야 합니다. 학습되는 두 벡터 $w_{\text{start}}, w_{\text{end}} \in \mathbb{R}^d$가 각 토큰의 점수를 줍니다.

$$P_{\text{start}}(i) = \frac{\exp(w_{\text{start}} \cdot h_i)}{\sum_j \exp(w_{\text{start}} \cdot h_j)}, \qquad P_{\text{end}}(i) = \frac{\exp(w_{\text{end}} \cdot h_i)}{\sum_j \exp(w_{\text{end}} \cdot h_j)}$$

예측된 답은 $i \le k$이면서 $P_{\text{start}}(i) \cdot P_{\text{end}}(k)$를 최대화하는 구간 $(i, k)$입니다.

## 코드: 작은 BERT 스타일 분류기

아래 모델에는 BERT의 모든 요소가 축소판으로 들어 있습니다. 세 가지 임베딩, 인과적 마스크 없는 인코더, 패딩 마스크, `[CLS]` 위의 분류 헤드입니다. 학습되지 않았으므로 출력은 무작위이지만, 텐서의 형태와 데이터 흐름은 실제와 같습니다.

```python
import torch
import torch.nn as nn

class TinyBERT(nn.Module):
    def __init__(self, vocab=1000, d=64, heads=4, layers=2, max_len=32, n_classes=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)        # 토큰 임베딩
        self.pos = nn.Embedding(max_len, d)      # 학습되는 위치 (제17장)
        self.seg = nn.Embedding(2, d)            # 세그먼트 A / 세그먼트 B
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, activation="gelu",
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)      # [CLS] 위의 분류 헤드

    def forward(self, ids, segs, pad_mask):
        pos = torch.arange(ids.size(1)).unsqueeze(0)
        h = self.tok(ids) + self.pos(pos) + self.seg(segs)
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # 인과적 마스크 없음!
        return self.head(h[:, 0])                # 위치 0의 벡터가 [CLS]

PAD, CLS, SEP = 0, 1, 2
# "[CLS] a b [SEP] c d e [SEP] [PAD]"
ids  = torch.tensor([[CLS, 15, 27, SEP, 40, 41, 42, SEP, PAD]])
segs = torch.tensor([[0,   0,  0,  0,   1,  1,  1,  1,   0]])
pad_mask = ids == PAD                            # True는 무시할 패딩을 표시
model = TinyBERT()
print(model(ids, segs, pad_mask).shape)          # torch.Size([1, 2])
```

미세 조정하려면 이 로짓과 정답 레이블 사이의 `F.cross_entropy`를 계산하고 제10장의 학습 루프를 돌리면 됩니다. 토큰 분류를 하려면 `h[:, 0]`을 `h`로 바꾸어 토큰마다 예측 하나를 얻습니다.

## BERT 계열과 오늘날 인코더가 쓰이는 곳

BERT 이후 많은 개선된 인코더가 등장했습니다. 이름을 알아 둘 만한 것은 다음과 같습니다.

- **RoBERTa** — 같은 아키텍처, 더 나은 학습 방법(위 참조).
- **ALBERT** — 층 사이에서 파라미터를 공유하고 임베딩 행렬을 인수분해하여 메모리상의 모델 크기를 크게 줄입니다.
- **DistilBERT** — BERT의 출력을 흉내 내도록 **지식 증류**(knowledge distillation)로 학습한 더 작은 학생 모델입니다. 더 빠르면서 품질의 대부분을 유지합니다.
- **ELECTRA** — 대체 토큰 탐지(제21장)로 학습하므로 손실이 모든 토큰을 포함합니다.
- **DeBERTa** — 어텐션 안에서 내용과 위치를 분리하여 표현합니다(*disentangled attention*).
- **mBERT**와 **XLM-R** — 러시아어와 한국어를 포함한 여러 언어로 한꺼번에 학습한 다국어 인코더입니다(제30장).

큰 디코더 모델이 거의 모든 것을 할 수 있는데 왜 인코더가 여전히 쓰일까요? 주된 이유는 비용과 과제의 성격, 두 가지입니다. 파라미터가 1억 개인 모델은 평범한 하드웨어에서 하루에 수백만 개의 문서를 분류할 만큼 저렴합니다. 필터링, 태깅, 라우팅, 모더레이션 같은 많은 산업 과제는 생성이 필요 없는 읽기 과제입니다.

가장 중요한 현대적 용도는 **의미 검색**(semantic search)입니다. 인코더의 출력 벡터를 평균하거나(*mean pooling*) $h_{\text{[CLS]}}$를 취하면 문장마다 하나의 벡터를 얻습니다. Sentence-BERT(Reimers and Gurevych, 2019)처럼 **대조 학습**(contrastive learning)으로 미세 조정하면, 이런 모델은 의미가 비슷한 문장을 가까운 벡터로 보냅니다. 그러면 검색은 코사인 유사도(제5장)로 최근접 이웃을 찾는 문제로 귀결됩니다. 이것이 제37장 검색 증강 생성의 '검색' 부분입니다.

인코더가 잘하지 못하는 것은 바로 글쓰기입니다. 모든 토큰이 다른 모든 토큰을 보므로 텍스트를 만들어 낼 자연스러운 왼쪽→오른쪽 순서가 없고, MLM은 모델에게 시퀀스 전체를 생성하도록 가르친 적이 없습니다. 생성을 위해서는 다음 장의 디코더 전용 모델이 필요합니다.

## 요약

- BERT는 양방향 어텐션을 쓰는 트랜스포머 인코더 블록을 쌓은 것입니다. BERT-base는 12개 층과 약 1억 1천만 개의 파라미터를 가집니다.
- 입력은 `[CLS]`, 세그먼트 A, `[SEP]`, 세그먼트 B, `[SEP]`이며, 각 입력 벡터는 토큰·위치·세그먼트 임베딩의 합입니다.
- BERT는 MLM과 다음 문장 예측으로 사전 학습되었습니다. RoBERTa는 NSP가 필요 없으며 학습을 잘하는 것이 매우 중요함을 보였습니다.
- 미세 조정은 작은 헤드를 추가합니다. 분류에는 `[CLS]` 위에, 태깅에는 모든 토큰 위에, 추출형 질의응답에는 시작/끝 벡터를 둡니다.
- 인코더는 분류와 의미 검색에서 저렴하고 강력한 선택으로 남아 있지만, 텍스트 생성을 위해 설계되지는 않았습니다.

## 연습 문제

1. `[CLS]` 토큰 자체는 아무 의미도 없는데도 그 출력 벡터가 입력 전체의 요약 역할을 할 수 있는 이유를 자신의 말로 설명하십시오.
2. 어휘 30,000개와 $d = 768$에 대해 BERT-base의 토큰 임베딩 행렬 크기를 추정하십시오. 약 1억 1천만 파라미터 중 몇 퍼센트입니까?
3. `TinyBERT`를 레이블 5개의 토큰 분류기로 바꾸고, 예제 입력의 출력 형태가 `(1, 9, 5)`인지 확인하십시오.
4. 추출형 질의응답에서 세 토큰에 대해 $P_{\text{start}} = (0.1, 0.6, 0.3)$, $P_{\text{end}} = (0.2, 0.3, 0.5)$라고 합시다. $i \le k$인 어떤 구간 $(i, k)$가 선택되며 그 점수는 얼마입니까?

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 인코더 전용 모델 | encoder-only model | модель только с кодировщиком | modél' tól'ko s kodiróvshchikom |
| 특수 토큰 | special token | специальный токен | spetsiál'nyy tóken |
| 토크나이저 | tokenizer | токенизатор | tokenizátor |
| 세그먼트 임베딩 | segment embedding | эмбеддинг сегмента | embédding segménta |
| 패딩 | padding | дополнение | dopolnéniye |
| 패딩 마스크 | padding mask | маска паддинга | máska páddinga |
| 다음 문장 예측 | next sentence prediction | предсказание следующего предложения | predskazániye sléduyushchego predlozhéniya |
| 전이 학습 | transfer learning | перенос обучения | perenós obuchéniya |
| 미세 조정 | fine-tuning | дообучение | doobuchéniye |
| 분류 헤드 | classification head | классификационная голова | klassifikatsiónnaya golová |
| 개체명 인식 | named entity recognition | распознавание именованных сущностей | raspoznavániye imenóvannykh súshchnostey |
| 질의응답 | question answering | вопросно-ответная система | voprósno-otvétnaya sistéma |
| 지식 증류 | knowledge distillation | дистилляция знаний | distillyátsiya znániy |
| 대조 학습 | contrastive learning | контрастивное обучение | kontrastívnoye obuchéniye |
| 의미 검색 | semantic search | семантический поиск | semantícheskiy póisk |

---
[← 이전: 사전 학습 목표: 인과적 언어 모델링과 마스크 언어 모델링](21-pretraining-objectives.md) · [목차](README.md) · [다음: GPT와 디코더 전용 모델, 문맥 내 학습 →](23-gpt-and-decoders.md)
