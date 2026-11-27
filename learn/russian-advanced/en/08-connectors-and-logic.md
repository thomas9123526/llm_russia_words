# Lesson 8. Connectors and Logical Structure

> **Advanced Russian for Researchers** · Level: B2 · **You will learn:** connectors by function and where they take commas · cause, consequence, concession and contrast in complex sentences (поскольку, несмотря на то что, тогда как) · metatext: announcing the structure of a paper and guiding the reader

## Text

This is the introduction to the article whose method section you read in Lesson 7. Notice how almost every sentence begins with a word that tells the reader how it relates to the previous one.

**1. Введение**

Большие языковые модели (БЯМ) за последние годы стали основным инструментом автоматической обработки текстов. Они успешно решают задачи перевода, реферирования и ответа на вопросы. Однако качество их работы во многом зависит от языка: на английском языке модели, как правило, показывают более высокие результаты, чем на русском или корейском.

*In recent years large language models (LLMs) have become the main tool of automatic text processing. They successfully handle translation, summarization and question answering. However, the quality of their work depends heavily on the language: in English, models as a rule show higher results than in Russian or Korean.*

Причин этого несколько. Во-первых, доля английских текстов в обучающих данных значительно выше. Во-вторых, токенизаторы, построенные преимущественно на английских текстах, разбивают русские и корейские слова на большее число токенов. В результате последовательности становятся длиннее, а обработка — дороже. Кроме того, при одинаковом размере контекстного окна модель «видит» меньший фрагмент текста на этих языках.

*There are several reasons for this. First, the share of English texts in the training data is much higher. Second, tokenizers built mainly on English texts split Russian and Korean words into more tokens. As a result, sequences become longer and processing more expensive. In addition, with the same context window size, the model "sees" a smaller fragment of text in these languages.*

Существующие подходы к решению этой проблемы можно разделить на две группы. В работах первой группы предлагается расширять словарь токенизатора за счёт токенов целевого языка. Авторы работ второй группы, напротив, сохраняют словарь и дообучают модель на дополнительных одноязычных данных. Первый подход сокращает длину последовательностей, тогда как второй повышает качество понимания текста. Тем не менее ни один из них не решает проблему полностью: расширение словаря требует повторного обучения входного слоя, а дообучение никак не влияет на длину последовательностей.

*Existing approaches to this problem fall into two groups. Works of the first group propose extending the tokenizer's vocabulary with tokens of the target language. Authors of the second group, on the contrary, keep the vocabulary and fine-tune the model on additional monolingual data. The first approach shortens sequences, whereas the second improves text understanding. Nevertheless, neither solves the problem completely: extending the vocabulary requires retraining the input layer, while fine-tuning has no effect on sequence length.*

В связи с этим в данной работе предлагается комбинированный метод. Поскольку оба этапа можно выполнять последовательно, их преимущества, как мы предполагаем, суммируются. Иными словами, модель получает одновременно более короткие последовательности и лучшее представление целевого языка.

*For this reason, this paper proposes a combined method. Since both stages can be performed one after the other, their advantages, we assume, add up. In other words, the model gets both shorter sequences and a better representation of the target language.*

Вклад работы состоит в следующем. Во-первых, предложен комбинированный метод адаптации модели к новому языку. Во-вторых, проведено сравнение метода с двумя базовыми подходами на русском и корейском языках. Наконец, показано, что предложенный метод позволяет сократить вычислительные затраты без потери качества.

*The contribution of the work is as follows. First, a combined method for adapting a model to a new language is proposed. Second, the method is compared with two baseline approaches on Russian and Korean. Finally, it is shown that the proposed method reduces computational cost without loss of quality.*

Статья организована следующим образом. В разделе 2 рассматриваются связанные работы. В разделе 3 описывается предложенный метод, а в разделе 4 — постановка экспериментов. Результаты обсуждаются в разделе 5. В заключении подводятся итоги и намечаются направления дальнейших исследований.

*The paper is organised as follows. Section 2 reviews related work. Section 3 describes the proposed method, and Section 4 the experimental setup. The results are discussed in Section 5. The Conclusion sums up and outlines directions for future research.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| свя́зка | svyázka | connector, linking word (ж.) |
| во-пе́рвых, во-вторы́х | vo-pérvykh, vo-vtorýkh | firstly, secondly |
| бо́лее того́ | bólee togó | moreover, what is more |
| тем не ме́нее | tem ne ménee | nevertheless |
| вме́сте с тем | vméste s tem | at the same time, yet |
| тогда́ как | togdá kak | whereas |
| поско́льку | poskól'ku | since, because |
| сле́довательно | slédovatel'no | consequently, therefore |
| и́ными слова́ми | ínymi slovámi | in other words |
| напро́тив | naprótiv | on the contrary |
| всле́дствие (+ gen.) | vslédstvie | as a result of, owing to |
| несмотря́ на то что | nesmotryá na to chto | although, despite the fact that |
| в заключе́ние | v zaklyuchénie | in conclusion, finally |
| преиму́щественно | preimúshchestvenno | mainly, predominantly |
| реферирова́ние | referirovánie | summarization (ср.) |
| целево́й язы́к | tselevóy yazýk | target language (м.) |
| комбини́рованный | kombinírovannyy | combined |
| вычисли́тельные затра́ты | vychislítel'nye zatráty | computational cost (мн.) |
| постано́вка экспериме́нтов | postanóvka eksperiméntov | experimental setup (ж.) |
| намеча́ть / наме́тить | namechát' / namétit' | to outline, to sketch out |

## Grammar

A Russian scientific text is held together by **connectors** (связки): words and phrases that show whether the next sentence adds, contrasts, explains or concludes. They work like road signs. A reader who sees Однако knows a limitation is coming; Таким образом announces a conclusion. Using them correctly, and with the right punctuation, is one of the quickest ways to make your writing sound native.

### 1. Connectors by function

| Function | Connectors |
|---|---|
| addition | кроме того; более того (stronger: *what is more*); также, а также |
| contrast | однако; но; а; вместе с тем; в то же время; напротив; тогда как / в то время как; в отличие от + gen. |
| concession | тем не менее; несмотря на это; хотя; несмотря на то что |
| cause | так как; поскольку; благодаря тому что; из-за того что; в связи с тем что |
| consequence | поэтому; следовательно; в результате; в связи с этим; вследствие этого; так что |
| example, specification | например; в частности; так (at the start: *for instance*); а именно |
| reformulation | иными словами; другими словами; то есть |
| sequence | во-первых, во-вторых, в-третьих; наконец |
| conclusion | итак; таким образом; в заключение |

Some differences in meaning:

- **Однако** and **но** both mean *however / but*. Но links two parts of one sentence; Однако usually starts a new sentence and sounds more formal.
- **Тем не менее** is concession: *in spite of what was just said*. Данных было мало. Тем не менее модель обучилась хорошо.
- **Вместе с тем** and **в то же время** add a second, slightly contrasting side: Метод прост в реализации. Вместе с тем он требует большого объёма памяти.
- **Напротив** means *on the contrary*: the second fact is the opposite of the first.
- **Следовательно** is a logical conclusion (*it follows that*); **поэтому** is a plain *that is why*; **таким образом** sums up several preceding sentences.

**Commas.** Many connectors are introductory words (вводные слова, see Lesson 4) and are set off by commas. Others are conjunctions or adverbs and take no comma. This is a frequent source of errors:

| With commas (introductory) | Without commas |
|---|---|
| Во-первых, …; Во-вторых, …; Наконец, … (in a list) | Однако … (at the start of a sentence) |
| Кроме того, …; Более того, … | Тем не менее … |
| Например, …; В частности, … | Вместе с тем …; В то же время … |
| Иными словами, … | При этом … |
| Следовательно, …; Таким образом, …; Итак, … | Поэтому …; В результате …; В связи с этим … |
| Напротив, …; С одной стороны, … с другой стороны, … | Также …; В заключение … |

In the middle of a sentence однако **is** introductory and takes two commas: Модель, однако, допускает ошибки.

**В заключение or в заключении?** В заключение (with -е) is an adverb: *finally, to conclude*. В заключение отметим, что… В заключении (with -и) is a noun in the prepositional case: *in the Conclusion section*. В заключении статьи подводятся итоги. The last sentence of the text above uses the second one.

### 2. Cause, consequence, concession and contrast in complex sentences

Each logical relation has a conjunction (for a clause with a verb) and usually a preposition (for a noun phrase):

| Relation | Conjunction + clause | Preposition + noun |
|---|---|---|
| cause (neutral) | так как, поскольку | — |
| cause (formal) | в связи с тем что, ввиду того что, вследствие того что | в связи с + instr., ввиду + gen., вследствие + gen. |
| cause (positive) | благодаря тому что | благодаря + dat. |
| cause (negative) | из-за того что | из-за + gen. |
| concession | хотя, несмотря на то что | несмотря на + acc. |
| contrast | тогда как, в то время как | в отличие от + gen. |
| consequence | поэтому, так что, вследствие чего, в результате чего | — |

Examples:

- Поскольку данные размечены неточно, модель допускает ошибки. — *Since the data are annotated inaccurately, the model makes errors.*
- Благодаря тому что словарь был расширен, длина последовательностей сократилась. — *Thanks to the vocabulary being extended, sequence length decreased.*
- Несмотря на то что корпус небольшой, результаты оказались устойчивыми. — *Although the corpus is small, the results proved stable.*
- Первая модель обучалась две недели, тогда как вторая — три дня. — *The first model trained for two weeks, whereas the second took three days.*
- Объём памяти был ограничен, вследствие чего размер пакета пришлось уменьшить. — *Memory was limited, as a result of which the batch size had to be reduced.*

Notes:

- In written style **потому что** does not start a sentence; use **так как** or **поскольку** instead. Поскольку… can open the sentence; потому что… comes after the main clause.
- A compound conjunction usually has one comma before it: Модель ошибается, потому что данных мало. If you want to stress the cause, the comma moves inside: Модель ошибается потому, что данных мало (*precisely because*). The same with благодаря тому, что; из-за того, что.
- **Благодаря** is for positive results only. *Благодаря ошибке в данных точность снизилась* is wrong; use из-за.
- After **вследствие чего** and **в результате чего** the clause is joined with a comma, and чего refers back to the whole previous clause.

### 3. Metatext: guiding the reader

Metatext is text about the text. Russian papers use it at the end of the introduction, at the start of sections and in transitions:

| Purpose | Phrases |
|---|---|
| announcing the plan | Статья организована следующим образом. В разделе 2 рассматривается…, в разделе 3 описывается… |
| listing the contribution | Вклад работы состоит в следующем. Во-первых, … |
| looking back | Как было показано выше (в разделе 3), …; Как отмечалось ранее, … |
| looking ahead | Ниже будет показано, что…; Подробнее этот вопрос рассматривается в разделе 5. |
| changing topic | Перейдём к + dat.; Обратимся к + dat.; Рассмотрим + acc. |
| focusing | Остановимся подробнее на + prep. |
| summing up | Подведём итоги.; Таким образом, … |

A well-built Russian paragraph usually follows the same logic as the text above: a **topic sentence**, then **support** (во-первых, во-вторых, кроме того), then a **limitation** (однако, тем не менее, вместе с тем), then a **conclusion** (таким образом, следовательно). Look at the third paragraph of the text: two groups of approaches (topic), what each does (support), why neither is enough (limitation). The next paragraph opens with В связи с этим and draws the consequence.

## Study tip

Do not overuse connectors. One per sentence is plenty, and not every sentence needs one: when the logic is obvious, a plain sentence is better. Two habits mark a non-native text. The first is starting three sentences in a row with Также or Кроме того. The second is stacking two connectors with the same function (*Однако тем не менее…*). When you revise a draft, read only the first word of every sentence: you should see a varied, logical sequence (Однако… Во-первых… Кроме того… Таким образом…), not the same word repeated.

## Exercises

**Exercise 1.** Choose the right connector.
1. Модель обучалась на английских текстах. (Поэтому / Однако / Например) на русском языке она работает хуже.
2. Метод работает быстрее. (Кроме того / Тем не менее / Следовательно), он требует меньше памяти.
3. Первый подход сокращает длину последовательностей, (тогда как / поскольку / так что) второй повышает качество.
4. Данных было мало. (Тем не менее / Более того / Иными словами) модель показала хороший результат.
5. Токенизатор разбивает слово на много частей, (так что / хотя / в отличие от) последовательность становится длиннее.
6. Ряд моделей плохо работает с длинными текстами. (В частности / Наконец / Напротив), это относится к моделям с коротким контекстным окном.

**Exercise 2.** Add commas where they are needed. Some sentences need none.
1. Во-первых данные были собраны из открытых источников.
2. Однако модель допускает ошибки.
3. Модель однако допускает ошибки.
4. Тем не менее результаты оказались стабильными.
5. Следовательно метод можно применять и к другим языкам.
6. Поэтому мы использовали второй подход.
7. Иными словами модель видит меньше текста.
8. В заключение отметим основные результаты.

**Exercise 3.** Fill in в заключение or в заключении.
1. ___ статьи подводятся итоги.
2. ___ хотелось бы поблагодарить рецензентов.
3. Эти выводы повторяются в ___ диссертации.
4. ___ кратко перечислим ограничения работы.

**Exercise 4.** Combine the two sentences with the conjunction in brackets.
1. Модель допускает ошибки. Данные размечены неточно. (поскольку)
2. Словарь был расширен. Длина последовательностей сократилась. (благодаря тому что)
3. Корпус небольшой. Результаты оказались устойчивыми. (несмотря на то что)
4. Первая модель обучалась две недели. Вторая обучалась три дня. (тогда как)
5. Объём памяти был ограничен. Размер пакета пришлось уменьшить. (вследствие чего)

**Exercise 5.** Fill in the gaps. Use each preposition once: благодаря, из-за, несмотря на, в отличие от, в связи с.
1. ___ расширению словаря длина последовательностей сократилась.
2. ___ ошибки в разметке часть результатов пришлось пересчитать.
3. ___ небольшой объём корпуса, результаты оказались устойчивыми.
4. ___ предыдущих работ, мы оцениваем модель на двух языках.
5. ___ ограничением вычислительных ресурсов эксперименты проводились на модели меньшего размера.

**Exercise 6.** Put the sentences in the right order to make a paragraph.
- А. Кроме того, такие модели требуют меньше памяти.
- Б. Таким образом, несмотря на это ограничение, компактная модель является для нашей задачи более разумным выбором.
- В. Во-первых, компактные модели обучаются значительно быстрее.
- Г. Для решения поставленной задачи мы выбрали компактную модель по ряду причин.
- Д. Однако их качество на сложных вопросах несколько ниже.

**Exercise 7.** Answer the questions about the text.
1. Почему на русском и корейском языках модели работают хуже, чем на английском?
2. Чем различаются две группы существующих подходов?
3. Какой недостаток есть у каждого из двух подходов?
4. Что предлагается в данной работе?
5. Что делается в заключении статьи?

**Exercise 8.** Write two short paragraphs for a paper of your own: (1) the contribution of the work (Вклад работы состоит в следующем…) and (2) the plan of the paper (Статья организована следующим образом…). Use at least five different connectors or metatext phrases.

## Answer key

**Exercise 1.**
1. Поэтому
2. Кроме того
3. тогда как
4. Тем не менее
5. так что
6. В частности

**Exercise 2.**
1. Во-первых, данные были собраны из открытых источников.
2. Однако модель допускает ошибки. (без запятой)
3. Модель, однако, допускает ошибки.
4. Тем не менее результаты оказались стабильными. (без запятой)
5. Следовательно, метод можно применять и к другим языкам.
6. Поэтому мы использовали второй подход. (без запятой)
7. Иными словами, модель видит меньше текста.
8. В заключение отметим основные результаты. (без запятой)

**Exercise 3.**
1. В заключении
2. В заключение
3. заключении
4. В заключение

**Exercise 4.**
1. Модель допускает ошибки, поскольку данные размечены неточно. / Поскольку данные размечены неточно, модель допускает ошибки.
2. Благодаря тому что словарь был расширен, длина последовательностей сократилась. / Длина последовательностей сократилась благодаря тому, что словарь был расширен.
3. Несмотря на то что корпус небольшой, результаты оказались устойчивыми.
4. Первая модель обучалась две недели, тогда как вторая обучалась три дня. / Первая модель обучалась две недели, тогда как вторая — три дня.
5. Объём памяти был ограничен, вследствие чего размер пакета пришлось уменьшить.

**Exercise 5.**
1. Благодаря
2. Из-за
3. Несмотря на
4. В отличие от
5. В связи с

**Exercise 6.**
Г – В – А – Д – Б

**Exercise 7.**
1. Во-первых, доля английских текстов в обучающих данных значительно выше. Во-вторых, токенизаторы разбивают русские и корейские слова на большее число токенов, поэтому последовательности длиннее, а модель видит меньший фрагмент текста.
2. В работах первой группы расширяют словарь токенизатора, а авторы второй группы сохраняют словарь и дообучают модель на одноязычных данных.
3. Расширение словаря требует повторного обучения входного слоя, а дообучение не влияет на длину последовательностей.
4. Комбинированный метод, который объединяет расширение словаря и дообучение.
5. Подводятся итоги и намечаются направления дальнейших исследований.

**Exercise 8.** Model answer:

Вклад работы состоит в следующем. Во-первых, собран корпус научных текстов на русском языке. Во-вторых, предложен метод фильтрации, который учитывает долю формул в документе. Кроме того, проведено сравнение трёх токенизаторов на этом корпусе. Наконец, показано, что фильтрация позволяет сократить время обучения без потери качества.

Статья организована следующим образом. В разделе 2 рассматриваются связанные работы. В разделе 3 описывается корпус, а в разделе 4 — метод фильтрации. Результаты экспериментов обсуждаются в разделе 5. В заключении подводятся итоги и обсуждаются ограничения работы.

Checklist:
- the contribution is a numbered list: во-первых, во-вторых, (кроме того,) наконец;
- the contribution items use short passive forms (предложен, проведено, показано);
- the plan names each section with a verb in -ся (рассматриваются, описывается, обсуждаются);
- ellipsis with a dash (а в разделе 4 — …) to avoid repeating the verb;
- commas after introductory connectors and no commas after однако / тем не менее / в заключение at the start of a sentence;
- в заключении (the section) vs в заключение (finally) spelled correctly.

## Review

- You can choose a connector by its function: addition, contrast, concession, cause, consequence, example, reformulation, sequence, conclusion.
- You know which connectors take commas (во-первых, кроме того, следовательно, таким образом, иными словами) and which do not (однако at the start, тем не менее, поэтому, в результате, в заключение).
- You can tell в заключение from в заключении.
- You can link clauses with поскольку, благодаря тому что, несмотря на то что, тогда как, вследствие чего, and you know the matching prepositions with their cases.
- You can write the "contribution" and "structure of the paper" paragraphs that end a Russian introduction.

---
[← Previous: Nominal Style: Verbal Nouns and Noun Chains](07-nominal-style.md) · [Contents](README.md) · [Next: Hedging, Certainty and Evaluation →](09-hedging-and-evaluation.md)
