# 제20장. PyTorch로 작은 트랜스포머 만들기

> **제4부 — 트랜스포머** · 수준: 중급
>
> **이 장에서 배울 내용:** 임베딩, 어텐션 블록, 출력 헤드를 조립해 동작하는 GPT 스타일 모델을 만드는 방법 · 다음 토큰 예측으로 학습시키고 손실 곡선을 읽는 방법 · 모델로 텍스트를 생성하는 방법과 이 장난감 모델이 실제 LLM과 무엇이 다른가

## 파라미터 100,000개로 글쓰기를 배울 수 있을까?

제16–19장에서 우리는 트랜스포머를 부품별로 공부했습니다. 인과 마스크를 쓰는 멀티헤드 어텐션, 위치 정보, 잔차 연결·정규화·FFN을 갖춘 블록, 그리고 디코더 전용 아키텍처입니다. 이제 모든 것을 모아 구체적인 질문을 던져 봅시다. 파라미터가 약 100,000개인 모델을 노트북 CPU에서 15초 동안 학습시키면, 러시아어 문장을 한 글자씩 만들어 내는 법을 배울 수 있을까요?

우리는 GPT의 정신을 따르는 아주 작은 디코더 전용 모델을 PyTorch 약 65줄로 만들겠습니다. 이름은 **TinyGPT**라고 합시다. 프로그램은 아래에 데이터, 모델, 학습, 생성의 네 부분으로 나뉘어 있습니다. **네 개의 코드 블록을 차례로 하나의 파일**(예: `tiny_gpt.py`)**에 넣고** `python tiny_gpt.py`로 실행하십시오. PyTorch만 있으면 되고 CPU에서 동작합니다. (Windows에서 키릴 문자를 출력할 때 인코딩 오류가 나면 먼저 환경 변수 `PYTHONIOENCODING=utf-8`을 설정하십시오.)

이 장난감 모델은 일부러 작게 만들었지만 *아이디어*를 단순화한 것은 아닙니다. 같은 부품을 몇 자릿수 더 키우고, 더 나은 토크나이저와 훨씬 많은 데이터를 쓰면 실제 LLM이 됩니다.

## 1단계: 데이터와 문자 토크나이저

모든 것을 눈에 보이게 하기 위해 **문자 수준 토크나이저(character-level tokenizer)**를 사용합니다. 서로 다른 문자 하나하나가 토큰 하나입니다. 제3장에서 보았듯이 실제 모델은 BPE 같은 서브워드 토크나이저를 쓰지만, 문자 어휘는 학습이 필요 없고 모델이 무엇을 배웠는지 보기 쉽습니다. 학습 텍스트는 러시아어 두 문장을 20번 반복한 것입니다. *«модель читает текст и предсказывает следующий символ»*(모델은 텍스트를 읽고 다음 문자를 예측한다)와 *«внимание помогает модели смотреть на прошлые символы»*(어텐션은 모델이 과거 문자를 보도록 돕는다)입니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
text = ("модель читает текст и предсказывает следующий символ. "
        "внимание помогает модели смотреть на прошлые символы. ") * 20
chars = sorted(set(text))                          # 문자 수준 어휘
stoi = {c: i for i, c in enumerate(chars)}         # 문자 -> 번호
itos = {i: c for c, i in stoi.items()}             # 번호 -> 문자
data = torch.tensor([stoi[c] for c in text])
V, T, D, H, L = len(chars), 32, 64, 4, 2           # 어휘, 문맥, 폭, 헤드, 층
```

**어휘(vocabulary)**는 키릴 문자, 공백, 마침표로 이루어진 $V = 26$개의 기호입니다. 텍스트 전체는 토큰 번호로 된 1차원 텐서 `data`가 됩니다. **하이퍼파라미터(hyperparameter)**는 일부러 아주 작게 잡았습니다. **문맥 창(context window)** $T = 32$문자, 모델 폭 $D = 64$, 헤드 $H = 4$개(따라서 각 헤드의 $d_k = 16$), 블록 $L = 2$개입니다. 비교하자면 GPT-2 small은 폭 768, 헤드 12개, 블록 12개입니다.

## 2단계: 모델

블록은 제18장의 Pre-Norm 형태를 따릅니다. LayerNorm → 인과적 멀티헤드 어텐션 → 잔차, 그다음 LayerNorm → FFN → 잔차입니다. 어텐션에는 이제 PyTorch 내장 함수 `F.scaled_dot_product_attention`을 씁니다. 이 함수는 제16장의 공식을 그대로 계산하며, `is_causal=True` 플래그가 인과 마스크를 대신 적용해 줍니다. FFN은 은닉 크기 $4D$와 GELU를 쓰는 고전적인 형태입니다.

```python
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(self, x):
        B, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = [z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # 마스크드 어텐션
        x = x + self.proj(y.transpose(1, 2).reshape(B, t, D))        # 잔차 연결 1
        return x + self.ffn(self.ln2(x))                             # 잔차 연결 2

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)              # 토큰 임베딩
        self.pos = nn.Embedding(T, D)              # 학습형 위치 (제17장)
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)                # 최종 정규화
        self.head = nn.Linear(D, V, bias=False)    # LM 헤드: 벡터 -> 로짓
        self.head.weight = self.tok.weight         # 가중치 공유
        nn.init.normal_(self.tok.weight, std=0.02) # 작은 초기화: 로짓이 0 근처에서 시작
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        return self.head(self.ln_f(self.blocks(x)))
```

전체 모델은 제18장 끝의 도식처럼 위에서 아래로 읽힙니다. 토큰 번호는 임베딩 표(제5장)를 거쳐 벡터가 되고, 학습형 위치 벡터가 더해지며, 블록 두 개가 시퀀스를 처리하고, 최종 LayerNorm이 적용됩니다. 마지막으로 크기 $D \to V$인 선형층인 **언어 모델 헤드(language model head)**가 각 벡터를 $V$개의 **로짓(logits)**, 즉 다음 토큰에 대한 정규화되지 않은 점수로 바꿉니다.

두 가지 세부 사항은 설명이 필요합니다.

**가중치 공유.** 헤드 행렬의 모양은 $V \times D$로, 임베딩 표와 똑같습니다. GPT-2를 비롯한 많은 모델이 둘에 *같은* 행렬을 씁니다. 이 **가중치 공유(weight tying)**는 파라미터를 절약하며 직관적인 의미도 있습니다. 토큰 $j$의 로짓은 최종 벡터와 $j$의 임베딩의 내적이므로, 모델은 임베딩이 출력과 같은 방향을 가리키는 토큰을 예측합니다.

**초기화.** PyTorch는 `nn.Embedding`을 표준편차 1로 초기화합니다. 가중치를 공유하면 이 때문에 초기 로짓이 커지고, 첫 손실이 기대값 $\ln V$가 아니라 약 45가 됩니다. 왜 $\ln V$일까요? 아무것도 모르는 모델은 모든 토큰에 확률 $1/V$를 주어야 하며, 이 추측의 교차 엔트로피는 $-\ln(1/V) = \ln 26 \approx 3.26$입니다. GPT-2처럼 임베딩을 작은 표준편차 0.02로 다시 **초기화(initialization)**하면 이 합리적인 출발점에 도달합니다. 첫 손실이 $\ln V$에 가까운지 확인하는 것은 모든 분류기에 대해 좋은 습관입니다.

모델의 파라미터는 **103,808개**입니다. 블록 하나에 $12{,}480$(QKV) $+ 4{,}160$(출력 투영) $+ 33{,}088$(FFN) $+ 256$(LayerNorm 두 개) $= 49{,}984$개가 있고, 블록 두 개면 $99{,}968$개입니다. 위치 표가 $32 \cdot 64 = 2{,}048$개, 토큰 표가 $26 \cdot 64 = 1{,}664$개(헤드와 공유), 최종 LayerNorm이 $128$개를 더합니다.

## 3단계: 다음 토큰 예측으로 학습하기

학습은 제8–10장의 루프입니다. 매 스텝마다 텍스트에서 $T + 1$문자짜리 무작위 창 16개를 잘라 냅니다. 처음 $T$문자가 입력 `x`이고, 한 위치 밀린 같은 창이 목표 `y`입니다. 인과 마스크 덕분에 창 하나가 한 번에 $T = 32$개의 학습 예제를 줍니다. 위치 0은 문자 1을 예측하고, 위치 1은 문자 0–1로 문자 2를 예측하는 식입니다. 손실은 로짓과 실제 다음 문자 사이의 **교차 엔트로피(cross-entropy)**(제8장)를 $16 \times 32$개 위치 전체에 대해 평균한 것이며, 옵티마이저는 AdamW(제9장)입니다.

```python
model = TinyGPT()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print(sum(p.numel() for p in model.parameters()), "parameters")

for step in range(1001):
    ix = torch.randint(len(data) - T - 1, (16,))             # 무작위 창 16개
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])     # 목표: 한 칸 밀림
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(step, round(loss.item(), 3))
```

우리 컴퓨터(PyTorch 2, CPU)에서는 다음과 같이 출력되었습니다.

```
103808 parameters
0 3.27
200 2.955
400 1.439
600 0.398
800 0.109
1000 0.088
```

정확한 숫자는 PyTorch 버전에 따라 조금 다를 수 있지만, 곡선의 모양은 많은 것을 알려 줍니다. 곡선은 $3.27 \approx \ln 26$, 즉 균등한 추측에서 시작합니다. 그다음 한동안 $2.95$ 근처에 머뭅니다. 이 값은 우연이 아닙니다. 우리 텍스트의 문자 빈도의 엔트로피가 $2.946$입니다. 이 단계에서 모델은 *어떤 문자가 흔한지*(공백, «о», «е»)는 배웠지만 아직 문맥을 활용하지 못합니다. 제4장의 의미에서 유니그램 모델인 셈입니다. 그러다 어텐션이 "켜지면" 손실이 빠르게 떨어지고, 1000스텝에서는 $0.1$ 아래로 내려갑니다. 모델이 거의 모든 다음 문자를 정확히 예측합니다. 이런 **정체 구간(plateau)** 뒤의 급격한 하락은 모델이 새로운 메커니즘, 여기서는 뒤를 돌아보고 복사하는 능력을 발견해야 할 때 흔히 나타납니다.

## 4단계: 텍스트 생성

생성은 제1장에서 설명한 대로 자기회귀적입니다. 모델에 프롬프트를 주고, **마지막** 위치의 로짓을 softmax로 확률로 바꾸고, 문자 하나를 **샘플링(sampling)**하여 덧붙이고, 이를 반복합니다. 위치 표에는 $T$개의 행만 있으므로 마지막 $T$문자까지만 넣습니다(`idx[:, -T:]`). 학습 중이 아니므로 데코레이터 `@torch.no_grad()`로 기울기 추적을 끕니다.

```python
@torch.no_grad()
def generate(prompt, n=60):
    idx = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(n):
        logits = model(idx[:, -T:])[:, -1, :]                # 마지막 위치만
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())

model.eval()
print(generate("модель "))
```

우리의 실행에서는 `модель читает текст и предсказывает следующий символ. внимание помо`가 출력되었습니다. 완벽한 이어짐입니다. 비교하자면 300스텝만 학습한 뒤(손실 약 2)에는 같은 코드가 `модель  пыотелюыемваей и пыом.уыи см.де` 같은 것을 출력했습니다. 러시아어처럼 보이는 글자가 올바른 빈도로 나오지만 단어는 없습니다. 프롬프트에는 어휘에 있는 문자만 쓸 수 있으며, 다른 문자가 있으면 `KeyError`가 납니다.

## TinyGPT와 실제 LLM의 차이

우리가 만든 것에 대해 솔직해집시다. 이 모델은 두 문장을 **암기(memorization)**했습니다. 텍스트를 20번 반복했으니 일반화할 것이 없고, 이를 측정할 검증 세트(제10장)도 따로 두지 않았습니다. 그래도 아키텍처는 진짜입니다. TinyGPT를 제5부에 나오는 모델 같은 것으로 바꾸려면 다음을 바꾸어야 합니다.

- **토크나이저:** 26개의 문자 대신 수만 개의 서브워드로 된 BPE나 SentencePiece 어휘(제3장).
- **데이터와 규모:** 다양한 텍스트 수십억 토큰, 그리고 몇 자릿수 더 큰 폭, 깊이, 헤드 수(제26–28장).
- **현대적 부품:** 학습형 위치 대신 RoPE(제17장), RMSNorm과 SwiGLU(제18장), 그리고 흔히 그룹 쿼리 어텐션(제25장).
- **학습 세부 사항:** 드롭아웃 등의 정규화, 학습률 워밍업과 감쇠, 기울기 클리핑, 혼합 정밀도(제10장, 제28장).
- **생성:** 온도, top-k·top-p 샘플링(제24장), 그리고 과거의 키와 값을 다시 계산하지 않기 위한 KV 캐시(제36장).

이 변화 중 어느 것도 방금 실행한 핵심 루프를 건드리지 않습니다. 임베딩하고, 어텐션하고, 변환하고, 다음 토큰을 예측하고, 교차 엔트로피를 계산하고, 업데이트합니다. 제5부는 바로 이 토대 위에 세워집니다.

## 요약

- GPT 스타일 모델은 토큰 임베딩 + 위치 → 인과적 트랜스포머 블록의 스택 → 최종 정규화 → 어휘에 대한 로짓을 내는 선형 헤드로 이루어집니다.
- 인과 마스크 덕분에 $T$토큰짜리 창 하나가 다음 토큰 예측 학습 예제 $T$개를 제공하며, 목표는 입력을 한 칸 민 것입니다.
- 잘 초기화된 모델은 손실 $\approx \ln V$에서 시작합니다. 유니그램 엔트로피 수준의 정체 구간은 모델이 문자 빈도는 알지만 아직 문맥을 쓰지 못함을 뜻합니다.
- 가중치 공유는 임베딩 행렬을 출력 헤드와 공유하여 파라미터를 절약합니다.
- 생성은 마지막 $T$토큰을 반복해서 넣고, 마지막 위치의 softmax에서 샘플링한 뒤, 결과를 덧붙이는 과정입니다.

## 연습 문제

1. 학습 목표 `y`가 입력 `x`를 한 위치 민 것인 이유와, 이것이 동작하려면 인과 마스크가 꼭 필요한 이유를 자신의 말로 설명해 보십시오.
2. 어휘가 50,000토큰일 때 기대되는 초기 손실을 계산해 보십시오. 첫 손실이 약 45라는 것이 왜 문제의 신호입니까?
3. `self.pos = ...` 줄과 `forward`의 위치 항을 지우고 다시 학습해 보십시오. 손실과 생성된 텍스트는 어떻게 됩니까? 제17장을 이용해 설명해 보십시오.
4. 학습 텍스트를 2,000자 이상의 실제 러시아어 또는 한국어 산문으로 바꾸고, 끝의 10%를 검증 세트로 남겨 200스텝마다 검증 손실을 출력해 보십시오. 모델은 언제부터 과적합하기 시작합니까?

## 핵심 용어

| 한국어 | English | Русский | 러시아어 발음 |
|---|---|---|---|
| 문자 수준 토크나이저 | character-level tokenizer | посимвольный токенизатор | posímvol'nyy tokenizátor |
| 어휘 | vocabulary | словарь | slovár' |
| 하이퍼파라미터 | hyperparameter | гиперпараметр | giperparámetr |
| 문맥 창 | context window | контекстное окно | kontékstnoye oknó |
| 언어 모델 헤드 | language model head | голова языковой модели | golová yazykovóy modéli |
| 로짓 | logits | логиты | logíty |
| 가중치 공유 | weight tying | связывание весов | svyázyvaniye vesóv |
| 초기화 | initialization | инициализация | initsializátsiya |
| 교차 엔트로피 | cross-entropy | перекрёстная энтропия | perekryóstnaya entropíya |
| 학습 루프 | training loop | цикл обучения | tsikl obuchéniya |
| 정체 구간 | plateau | плато | plató |
| 자기회귀 생성 | autoregressive generation | авторегрессионная генерация | avtoregressiónnaya generátsiya |
| 샘플링 | sampling | сэмплирование | semplírovaniye |
| 암기 | memorization | запоминание | zapominániye |

---
[← 이전: 인코더, 디코더, 인코더–디코더 아키텍처](19-transformer-architectures.md) · [목차](README.md) · [다음: 사전 학습 목표: 인과적 언어 모델링과 마스크 언어 모델링 →](21-pretraining-objectives.md)
