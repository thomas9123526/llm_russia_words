# Lesson 11. Reading a Full Research Paper: Strategy and Practice

> **Advanced Russian for Researchers** · Level: B2 · **You will learn:** three kinds of reading and the best order for reading the sections of a paper · how to tell facts, assumptions, limitations and plans apart by their marker phrases · how to turn a paper into Russian notes (конспект) and a short summary for a seminar

## Text

Ivan Petrovich asked Minho to read a paper and present it at the lab seminar next week. The paper below was written for this textbook: the authors, the method, the references and all numbers are invented examples. Read the Russian first, then check the translation after each section.

**А. В. Орлов, Н. С. Белова**

**Адаптивный ранний выход для ускорения генерации в больших языковых моделях**

**Аннотация.** В статье рассматривается проблема высокой вычислительной стоимости генерации текста большими языковыми моделями. Предлагается метод адаптивного раннего выхода, при котором модель прекращает вычисления на промежуточном слое, если её уверенность в следующем токене достаточно высока. Эксперименты на двух русскоязычных наборах данных показали, что метод сокращает время генерации в среднем на 35 % при снижении качества не более чем на 1 %.

*A. V. Orlov, N. S. Belova. Adaptive early exit for faster generation in large language models. Abstract. The paper considers the problem of the high computational cost of text generation by large language models. A method of adaptive early exit is proposed, in which the model stops computing at an intermediate layer if its confidence in the next token is high enough. Experiments on two Russian-language datasets showed that the method reduces generation time by 35% on average, with a drop in quality of no more than 1%.*

**1. Введение.** Большие языковые модели широко применяются для решения разнообразных задач, однако их использование требует значительных вычислительных ресурсов. При генерации каждого токена модель проходит через все слои, хотя для многих «простых» токенов, например знаков препинания или служебных слов, это, по-видимому, избыточно. Целью настоящей работы является разработка метода, позволяющего модели самостоятельно определять, сколько слоёв необходимо для каждого токена.

*1. Introduction. Large language models are widely used to solve a variety of tasks, but using them requires considerable computational resources. When generating each token, the model passes through all the layers, although for many "easy" tokens, such as punctuation marks or function words, this is apparently redundant. The aim of the present work is to develop a method that allows the model to decide by itself how many layers are needed for each token.*

**2. Обзор связанных работ.** Идея раннего выхода известна давно и применялась прежде всего в задачах классификации [1, 2]. В работах [3, 4] она была перенесена на генерацию текста, однако порог уверенности в них задаётся вручную и одинаков для всех токенов. В отличие от этих работ, мы предлагаем обучаемый порог.

*2. Related work. The idea of early exit has long been known and was used above all in classification tasks [1, 2]. In [3, 4] it was transferred to text generation, but there the confidence threshold is set by hand and is the same for all tokens. Unlike these works, we propose a trainable threshold.*

**3. Метод.** После каждого четвёртого слоя модели добавляется небольшой классификатор, оценивающий уверенность модели в следующем токене. Если эта оценка превышает порог, вычисления прекращаются и токен выдаётся сразу. Порог не фиксирован: он определяется отдельной небольшой сетью в зависимости от контекста. Обе дополнительные сети обучаются при замороженных параметрах основной модели, т. е. сама модель не изменяется.

*3. Method. After every fourth layer of the model, a small classifier is added that estimates the model's confidence in the next token. If this estimate exceeds the threshold, computation stops and the token is output at once. The threshold is not fixed: it is determined by a separate small network depending on the context. Both additional networks are trained with the parameters of the main model frozen, i.e. the model itself does not change.*

**4. Эксперименты.** Метод проверялся на открытой модели с 7 млрд параметров. Для оценки использовались два набора данных: вопросы по научным статьям и краткие пересказы новостей. В качестве базовых методов рассматривались полная модель (без раннего выхода) и ранний выход с фиксированным порогом [3].

*4. Experiments. The method was tested on an open model with 7 billion parameters. Two datasets were used for evaluation: questions about scientific papers and short news summaries. The full model (without early exit) and early exit with a fixed threshold [3] were considered as baselines.*

**5. Результаты и обсуждение.** Как видно из табл. 2, предложенный метод сокращает время генерации на 35 % по сравнению с полной моделью, тогда как метод с фиксированным порогом — лишь на 20 %. Качество ответов при этом снижается незначительно (менее чем на 1 %). Следует, однако, отметить, что на длинных и сложных вопросах выигрыш во времени заметно меньше. Мы предполагаем, что это связано с более высокой долей «трудных» токенов в таких ответах.

*5. Results and discussion. As can be seen from Table 2, the proposed method reduces generation time by 35% compared with the full model, whereas the method with a fixed threshold reduces it by only 20%. The quality of the answers drops only slightly (by less than 1%). It should be noted, however, that on long and complex questions the time gain is noticeably smaller. We assume that this is due to the higher share of "hard" tokens in such answers.*

**6. Заключение.** Таким образом, обучаемый порог позволяет ускорить генерацию без существенной потери качества. Ограничением работы является то, что метод проверялся только на одной модели. В дальнейшем планируется исследовать модели большего размера и другие языки.

*6. Conclusion. Thus, a trainable threshold makes it possible to speed up generation without a significant loss of quality. A limitation of the work is that the method was tested on only one model. In future work, the authors plan to study larger models and other languages.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| просмо́тровое чте́ние | prosmótrovoe chténie | skimming, reading to decide if a text is relevant (ср.) |
| ознакоми́тельное чте́ние | oznakomítel'noe chténie | reading for the main idea (ср.) |
| изуча́ющее чте́ние | izucháyushchee chténie | detailed, study reading (ср.) |
| ра́нний вы́ход | ránniy výkhod | early exit (м.) |
| промежу́точный слой | promezhútochnyy sloy | intermediate layer (м.) |
| уве́ренность | uvérennost' | confidence (ж.) |
| поро́г | poróg | threshold (м.) |
| избы́точный | izbýtochnyy | redundant, excessive |
| служе́бное сло́во | sluzhébnoe slóvo | function word (ср.) |
| превыша́ть / превы́сить | prevyshát' / prevýsit' | to exceed |
| заморо́женный | zamorózhennyy | frozen (of parameters) |
| вы́игрыш | výigrysh | gain (м.) |
| до́ля | dólya | share, proportion (ж.) |
| ограниче́ние | ogranichénie | limitation (ср.) |
| суще́ственный | sushchéstvennyy | significant, substantial |
| по-ви́димому | po-vídimomu | apparently, seemingly |
| в дальне́йшем | v dal'néyshem | in future, later on |
| конспе́кт | konspékt | (reading) notes, summary (м.) |
| те́зисы | tézisy | key points, theses (pl.) |
| вклад | vklad | contribution (м.) |

## Grammar

This lesson is about a skill rather than a form: reading a whole paper in Russian efficiently. You already know the parts of an article (Intermediate, Lesson 18) and how to read an abstract (Intermediate, Lesson 20). Now the goal is to read 10–20 pages without translating every word, and to come out with clear notes.

### 1. Three kinds of reading and the order of reading

Russian language teaching traditionally distinguishes several kinds of reading. Three are useful for a researcher:

- **просмотровое чтение** — skimming (5 minutes). Question: is this paper relevant to me? Read the title, the abstract, the headings, the figures and the conclusion.
- **ознакомительное чтение** — reading for the main idea (20–30 minutes). Question: what did the authors do and find? Read the introduction and the results; skip technical details.
- **изучающее чтение** — detailed reading (an hour or more). Question: how exactly does it work, and could I reproduce it? Read everything, including the method and every number. Do this only for papers you will cite in detail or build on.

You do not have to read a paper from the first page to the last. A good order is:

1. title and abstract;
2. conclusion — the authors' own short version of the result;
3. figures and tables;
4. the last paragraph of the introduction — the aim and the contribution;
5. results and discussion;
6. method;
7. related work — when you need to place the paper among others.

Each section answers one question, and each has its typical phrases:

| Section | Question | Typical phrases (from the text) |
|---|---|---|
| Аннотация | What is it about, what is the main result? | В статье рассматривается…; Предлагается…; Эксперименты показали, что… |
| Введение | Why is it important, what is the aim? | однако…; Целью настоящей работы является… |
| Обзор связанных работ | What existed before, what is new? | известна давно; в работах [3, 4]; в отличие от этих работ |
| Метод | How does it work? | добавляется; определяется; обучается |
| Эксперименты | On what data, compared with what? | для оценки использовались; в качестве базовых методов рассматривались |
| Результаты и обсуждение | What was found, with what reservations? | как видно из табл. 2; тогда как; следует, однако, отметить |
| Заключение | Contribution, limitations, future work | таким образом; ограничением работы является; в дальнейшем планируется |

### 2. Fact, assumption, limitation, plan

When you retell a paper, you must not present the authors' guess as a proven fact. Russian papers mark the status of each statement clearly (see Lesson 9 on hedging):

| Status | Markers | Example from the text |
|---|---|---|
| result, fact | показали, что; установлено; как видно из; present tense of the verb | метод сокращает время генерации на 35 % |
| assumption | по-видимому; мы предполагаем, что; вероятно; это может быть связано с | Мы предполагаем, что это связано с… |
| limitation, reservation | следует, однако, отметить; ограничением является; лишь; только | Ограничением работы является то, что… |
| plan | в дальнейшем планируется; представляется перспективным | В дальнейшем планируется исследовать… |
| contribution, contrast | в отличие от; впервые; тогда как | В отличие от этих работ, мы предлагаем… |

**Two reading traps.**

- **The dash for a missing verb.** In *…тогда как метод с фиксированным порогом — лишь на 20 %* the dash replaces the repeated verb. Restore it in your head: …тогда как метод с фиксированным порогом [сокращает время генерации] лишь на 20 %.
- **The subject after the verb.** In passive sentences with -ся, the subject usually stands after the verb: *После каждого четвёртого слоя модели добавляется небольшой классификатор…* Find the verb first (добавляется), then ask "what is added?" — классификатор. Then put the participial phrase (оценивающий…) aside. This is the method from Lesson 2, applied at reading speed.

### 3. Taking notes: from the paper to your конспект

A **конспект** is a short written summary of a text in your own words; **тезисы** are its key points as a list. When you turn a paper into notes, three things change:

- **мы → авторы.** *Мы предлагаем обучаемый порог* → *Авторы предлагают обучаемый порог.*
- **Impersonal -ся → active with авторы** (optional, but clearer): *Предлагается метод…* → *Авторы предлагают метод…*; *Метод проверялся на модели…* → *Авторы проверяли метод на модели…* (the imperfective is kept).
- **Your own evaluation is separated** from the content: first what the authors did, then what you think.

Useful phrases for notes and for a seminar summary:

- Статья посвящена + dat. / В статье рассматривается + nom.
- Авторы предлагают / описывают / сравнивают + acc.
- Метод основан на + prep.
- Основной результат состоит в том, что…
- Согласно результатам, …
- К достоинствам работы можно отнести + acc.; К недостаткам работы можно отнести + acc.
- Открытым остаётся вопрос о том, …

**Abbreviations** you will meet in almost every paper:

| Abbreviation | Full form | Meaning |
|---|---|---|
| т. е. | то есть | that is, i.e. |
| т. к. | так как | since, because |
| и т. д. | и так далее | etc. |
| в т. ч. | в том числе | including |
| напр. | например | for example |
| см. | смотри(те) | see |
| ср. | сравни(те) | compare, cf. |
| табл. / рис. | таблица / рисунок | table / figure |
| млн / млрд | миллион / миллиард | million / billion |

## Study tip

Make a one-page card in Russian for every paper you read: the reference; the problem in one sentence; the method in one sentence; the data; the main number; one limitation; your comment. After a semester you will have a small Russian-language literature database — and the material for your literature review (Lesson 12).

## Exercises

**Exercise 1.** In which section of the paper can you find this information? Write the number and name of the section.
1. Где авторы объясняют, чем их работа отличается от предыдущих?
2. Где описано, как устроен классификатор?
3. Где названы наборы данных?
4. Где авторы говорят об ограничении своей работы?
5. Где сформулирована цель работы?
6. Где приводятся цифры из таблицы 2?

**Exercise 2.** Answer the questions about the text.
1. Какую проблему решают авторы?
2. Когда модель прекращает вычисления?
3. Чем предложенный метод отличается от работ [3, 4]?
4. На сколько процентов метод сокращает время генерации по сравнению с полной моделью?
5. На каких вопросах выигрыш во времени меньше?
6. Что авторы планируют сделать в дальнейшем?

**Exercise 3.** Find the grammatical core of each sentence: the subject and the predicate.
1. После каждого четвёртого слоя модели добавляется небольшой классификатор, оценивающий уверенность модели в следующем токене.
2. Для оценки использовались два набора данных.
3. Целью настоящей работы является разработка метода, позволяющего модели самостоятельно определять, сколько слоёв необходимо для каждого токена.
4. В качестве базовых методов рассматривались полная модель и ранний выход с фиксированным порогом.
5. Идея раннего выхода известна давно и применялась прежде всего в задачах классификации.
6. Качество ответов при этом снижается незначительно.

**Exercise 4.** Write the full form of each abbreviation.
1. т. е.
2. т. к.
3. и т. д.
4. табл.
5. рис.
6. см.
7. ср.
8. в т. ч.
9. млрд
10. напр.

**Exercise 5.** What is the status of each statement? Write результат, предположение, ограничение, план or цель.
1. Метод сокращает время генерации на 35 %.
2. Мы предполагаем, что это связано с более высокой долей «трудных» токенов.
3. В дальнейшем планируется исследовать модели большего размера.
4. Целью настоящей работы является разработка метода.
5. Для многих «простых» токенов это, по-видимому, избыточно.
6. Метод проверялся только на одной модели.

**Exercise 6.** Rewrite each sentence for your notes, with авторы as the subject.
1. Мы предлагаем обучаемый порог.
2. Мы предполагаем, что это связано с долей «трудных» токенов.
3. В статье рассматривается проблема стоимости генерации.
4. Предлагается метод адаптивного раннего выхода.
5. Метод проверялся на модели с 7 млрд параметров.

**Exercise 7.** Верно или неверно? Correct the false statements.
1. Идея раннего выхода впервые предложена в этой статье.
2. В работах [3, 4] порог задаётся вручную.
3. Классификатор добавляется после каждого слоя.
4. При обучении дополнительных сетей параметры основной модели не изменяются.
5. Метод с фиксированным порогом сокращает время генерации на 35 %.
6. Качество ответов снижается менее чем на 1 %.

**Exercise 8.** Write a summary of the paper for the lab seminar (6–8 sentences): topic, method, what is new, data, main results, one limitation, your opinion. Use phrases from Grammar section 3.

## Answer key

**Exercise 1.**
1. Раздел 2, «Обзор связанных работ».
2. Раздел 3, «Метод».
3. Раздел 4, «Эксперименты».
4. Раздел 6, «Заключение».
5. Раздел 1, «Введение».
6. Раздел 5, «Результаты и обсуждение».

**Exercise 2.**
1. Проблему высокой вычислительной стоимости генерации текста большими языковыми моделями.
2. Если её уверенность в следующем токене превышает порог. (*also:* Если её уверенность в следующем токене достаточно высока.)
3. В нём порог не задаётся вручную, а обучается: его определяет отдельная сеть в зависимости от контекста.
4. На 35 %.
5. На длинных и сложных вопросах.
6. Исследовать модели большего размера и другие языки.

**Exercise 3.**
1. классификатор добавляется
2. два набора использовались
3. разработка является целью
4. модель и ранний выход рассматривались
5. идея известна и применялась
6. качество снижается

**Exercise 4.** 1. то есть 2. так как 3. и так далее 4. таблица 5. рисунок 6. смотри (смотрите) 7. сравни (сравните) 8. в том числе 9. миллиард 10. например

**Exercise 5.** 1. результат 2. предположение 3. план 4. цель 5. предположение 6. ограничение

**Exercise 6.**
1. Авторы предлагают обучаемый порог.
2. Авторы предполагают, что это связано с долей «трудных» токенов.
3. Авторы рассматривают проблему стоимости генерации.
4. Авторы предлагают метод адаптивного раннего выхода.
5. Авторы проверяли метод на модели с 7 млрд параметров.

**Exercise 7.**
1. Неверно: идея раннего выхода известна давно.
2. Верно.
3. Неверно: классификатор добавляется после каждого четвёртого слоя.
4. Верно.
5. Неверно: метод с фиксированным порогом сокращает время генерации лишь на 20 %.
6. Верно.

**Exercise 8.** Model answer:
Статья А. В. Орлова и Н. С. Беловой посвящена ускорению генерации в больших языковых моделях. Авторы предлагают метод адаптивного раннего выхода: модель прекращает вычисления на промежуточном слое, если она достаточно уверена в следующем токене. В отличие от предыдущих работ, порог уверенности не задаётся вручную, а определяется отдельной сетью. Метод проверялся на модели с 7 млрд параметров и на двух русскоязычных наборах данных. Согласно результатам, время генерации сокращается на 35 %, а качество снижается менее чем на 1 %. Однако на длинных и сложных вопросах выигрыш заметно меньше. К недостаткам работы можно отнести то, что метод проверялся только на одной модели. На мой взгляд, было бы интересно проверить метод на моделях большего размера.

Checklist: the topic is named with посвящена + dative; the authors, not "мы", are the subject; what is new is stated with в отличие от; the main number is given correctly (на 35 %, менее чем на 1 %); the authors' assumptions are not presented as facts; one limitation is mentioned; your own opinion is separated from the content (На мой взгляд, …).

## Review

- You can choose the kind of reading you need: просмотровое, ознакомительное or изучающее.
- You read a paper in a smart order: abstract, conclusion, figures, aim, results, method.
- You recognise what each section answers and its typical phrases.
- You can tell results, assumptions, limitations and plans apart by their markers.
- You can restore a verb replaced by a dash and find the subject after a -ся verb.
- You can write a конспект with авторы as the subject and give a short summary at a seminar.

---
[← Previous: Typical Mistakes of Foreign Writers and How to Fix Them](10-typical-mistakes.md) · [Contents](README.md) · [Next: Writing a Literature Review (обзор литературы) →](12-literature-review.md)
