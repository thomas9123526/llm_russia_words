# Lesson 20. Reading an LLM Paper Abstract in Russian

> **Intermediate Russian** · Level: B1 · **You will learn:** the typical moves of a Russian abstract (аннотация) and their marker phrases · a strategy for reading long scientific sentences · numbers in results: на 12 %, в сто раз, с … до …

## Text

Minho is preparing for his first Russian-language conference. Ivan Petrovich gives him a model abstract to study. (The abstract below was written for this textbook; the method, dataset and numbers are invented.)

**Эффективное дообуче́ние большой языковой модели для ответов на вопросы по научным текстам**

**Аннота́ция.** Большие языковые модели (БЯМ) показывают высокие результаты во многих задачах обрабо́тки есте́ственного языка. Однако ка́чество их работы с научными текстами на русском языке остаётся недоста́точно изу́ченным. В настоящей работе предлагается метод дообучения языковой модели на небольшом ко́рпусе научных статей. Метод основан на сочета́нии ада́птеров низкого ранга (LoRA) и автомати́ческой генера́ции пар «вопрос — ответ». Для оценки был создан новый набор данных, содержа́щий 5 000 вопросов по статьям из области информатики. Эксперименты показали, что предложенный подход повыша́ет точность ответов на 12 % по сравнению с исхо́дной моделью, при этом число обуча́емых пара́метров сокраща́ется более чем в сто раз. Кроме того, было установлено, что модель реже генерирует недостове́рные факты (так называемые галлюцина́ции). Полученные результаты позволяют сделать вы́вод, что адаптация БЯМ к узкой предме́тной о́бласти возможна даже при ограниченных вычисли́тельных ресу́рсах.

**Ключевы́е слова́:** большие языковые модели, дообучение, LoRA, ответы на вопросы, научные тексты.

*Translation:*

*Efficient fine-tuning of a large language model for answering questions about scientific texts*

*Abstract. Large language models (LLMs) show high results in many natural language processing tasks. However, the quality of their work with scientific texts in Russian remains insufficiently studied. The present paper proposes a method for fine-tuning a language model on a small corpus of scientific articles. The method is based on a combination of low-rank adapters (LoRA) and automatic generation of question–answer pairs. For evaluation, a new dataset was created containing 5,000 questions about articles in computer science. The experiments showed that the proposed approach increases answer accuracy by 12% compared with the original model, while the number of trainable parameters is reduced more than a hundredfold. In addition, it was found that the model generates unreliable facts (so-called hallucinations) less often. The results obtained allow us to conclude that adapting LLMs to a narrow subject domain is possible even with limited computational resources.*

*Keywords: large language models, fine-tuning, LoRA, question answering, scientific texts.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| аннота́ция | annotátsiya | abstract (of a paper) (ж.) |
| дообуче́ние | doobuchénie | fine-tuning (ср.) |
| обрабо́тка есте́ственного языка́ | obrabótka estéstvennogo yazyká | natural language processing (ж.) |
| ка́чество | káchestvo | quality (ср.) |
| недоста́точно изу́ченный | nedostátochno izúchennyy | insufficiently studied |
| ко́рпус | kórpus | corpus (of texts) (м.) |
| сочета́ние | sochetánie | combination (ср.) |
| ада́птер | adápter | adapter (м.) |
| автомати́ческий | avtomatícheskiy | automatic |
| генера́ция | generátsiya | generation (ж.) |
| содержа́щий | soderzháshchiy | containing |
| повыша́ть | povyshát' | to increase, to raise |
| исхо́дный | iskhódnyy | original, initial |
| обуча́емый | obucháemyy | trainable |
| пара́метр | parámetr | parameter (м.) |
| сокраща́ться | sokrashchát'sya | to be reduced, to decrease |
| недостове́рный | nedostovérnyy | unreliable, not factual |
| галлюцина́ция | gallyutsinátsiya | hallucination (ж.) |
| вычисли́тельные ресу́рсы | vychislítel'nye resúrsy | computational resources (pl.) |
| предме́тная о́бласть | predmétnaya óblast' | subject domain (ж.) |
| вы́вод | vývod | conclusion (м.) |
| ключевы́е слова́ | klyuchevýe slová | keywords (pl.) |

## Grammar

### 1. The moves of an abstract and their marker phrases

Almost every abstract — Russian or English — makes the same five or six "moves" in the same order. In Lesson 18 you saw the parts of a whole article; an abstract is the whole article in miniature. If you know the typical phrases for each move, you can find the key information in seconds, even before you understand every word.

| Move | What it says | Typical Russian phrases |
|---|---|---|
| 1. Context | why the area matters | В последние годы… · …показывают высокие результаты · …широко используются |
| 2. Problem (gap) | what is still unknown | Однако… остаётся недостаточно изученным · …остаётся открытым вопросом |
| 3. Aim / method | what the paper does | В настоящей работе предлагается… · Метод основан на… · Целью работы является… |
| 4. Data | what was used | Для оценки был создан… · Эксперименты проводились на… |
| 5. Results | numbers and findings | Эксперименты показали, что… · Было установлено, что… |
| 6. Conclusion | what it means | Полученные результаты позволяют сделать вывод, что… · Таким образом… |

The word **однако** (*however*) is your best friend: it almost always marks the moment where the authors move from context to the problem they are solving. The phrase **в настоящей работе** (*in the present paper*) marks the authors' own contribution.

Notice the verb **остаётся** + instrumental: качество остаётся недостаточно изученным — *quality remains insufficiently studied*. Like являться and становиться, the verb оставаться takes the instrumental: остаётся открытым, остаётся важной задачей.

Look also at **основан на** + prepositional: метод основан на сочетании… — *the method is based on a combination of…*. Do not confuse it with на основе + genitive (Lesson 19): метод основан на сочетании = метод на основе сочетания.

### 2. How to read a long scientific sentence

Russian scientific sentences are long because they pack nouns into **genitive chains** and add **participle phrases**. Do not read them word by word from left to right. Use this four-step method:

1. **Find the verb.** Look for a finite verb: показали, повышает, сокращается, был создан.
2. **Find its subject** in the nominative: эксперименты, подход, число.
3. **Find the object or complement**: точность ответов, новый набор данных.
4. **Attach the rest**: prepositional phrases, participles, numbers.

Let us analyse two sentences from the text.

*Для оценки был создан новый набор данных, содержащий 5 000 вопросов по статьям из области информатики.*

- verb: был создан (*was created*); subject: новый набор данных
- the participle содержащий (masculine nominative) agrees with набор: *a dataset that contains 5,000 questions*
- по статьям — *on (about) articles*; из области информатики — *from the field of computer science*

*Эксперименты показали, что предложенный подход повышает точность ответов на 12 % по сравнению с исходной моделью, при этом число обучаемых параметров сокращается более чем в сто раз.*

- main clause: эксперименты показали — *the experiments showed*
- что-clause: подход повышает точность ответов — *the approach increases answer accuracy*
- на 12 % — by how much; по сравнению с исходной моделью — compared with what
- **при этом** — *at the same time, while*: it introduces an additional fact
- second clause: число обучаемых параметров сокращается — *the number of trainable parameters decreases*

**Genitive chains** are read from left to right as "X of Y of Z": метод дообучения языковой модели = *a method of fine-tuning of a language model* → *a method for fine-tuning a language model*. In Korean the order is reversed: 언어 모델의 추가 학습 방법. In English you often turn it into a compound: *language-model fine-tuning method*.

**Abbreviations.** Russian papers often use abbreviations: БЯМ (большая языковая модель), ИИ (искусственный интеллект), ОЕЯ (обработка естественного языка). They are introduced in brackets the first time: Большие языковые модели (БЯМ)… Many Russian authors also simply write the English abbreviation LLM.

### 3. Numbers in results: на, в … раз, с … до

The results move is full of numbers. The preposition tells you **what kind** of change it is:

| Construction | Meaning | Example |
|---|---|---|
| на + number | by (difference) | точность выросла на 6 % — *accuracy grew by 6%* |
| в + number + раз | times (ratio) | сократилось в три раза — *was reduced threefold* |
| с … до … | from … to … | выросла с 85 % до 91 % — *grew from 85% to 91%* |
| до + number | to (final value) | уменьшилось до 10 % — *decreased to 10%* |
| более чем / менее чем | more / less than | более чем в сто раз — *more than a hundredfold* |

Verbs of change come in pairs: повышать / повышаться (*to raise / to rise*), увеличивать / увеличиваться (*to increase*), сокращать / сокращаться (*to reduce / to decrease*), уменьшать / уменьшаться (*to diminish*). The form without -ся has an object (подход повышает точность); the form with -ся does not (точность повышается).

Comparative adverbs are frequent too: модель реже генерирует ошибки — *the model generates errors less often*; работает в два раза быстрее — *works twice as fast*.

Read the percent sign as процент in the right form (Lesson 16): на 12 % = на двенадцать процентов; на 1 % = на один процент; на 3 % = на три процента.

## Study tip

Russian journals usually print an аннотация and ключевые слова at the start of every article, and often an English version of both. When you search for literature, use this: read the English and Russian abstracts side by side, and you will learn the Russian terms of your field very quickly. When reading a new abstract, first read only the sentence with **однако** and the sentence with **показали** or **установлено**. Those two sentences tell you the problem and the result; decide after that whether the whole paper is worth reading.

## Exercises

**Exercise 1.** Answer the questions about the text.
1. Какая проблема описывается в аннотации?
2. Что предлагается в работе?
3. На чём основан метод?
4. Сколько вопросов содержит новый набор данных?
5. На сколько процентов повышается точность ответов?
6. Во сколько раз сокращается число обучаемых параметров?
7. Какой вывод делают авторы?

**Exercise 2.** Which move of the abstract is each sentence? Write A (context), B (problem), C (method), D (result) or E (conclusion).
1. Большие языковые модели (БЯМ) показывают высокие результаты во многих задачах обработки естественного языка.
2. Однако качество их работы с научными текстами на русском языке остаётся недостаточно изученным.
3. Метод основан на сочетании адаптеров низкого ранга (LoRA) и автоматической генерации пар «вопрос — ответ».
4. Эксперименты показали, что предложенный подход повышает точность ответов на 12 %.
5. Полученные результаты позволяют сделать вывод, что адаптация БЯМ к узкой предметной области возможна даже при ограниченных вычислительных ресурсах.

**Exercise 3.** Верно или неверно? (True or false?)
1. Метод требует очень больших вычислительных ресурсов.
2. Набор данных содержит вопросы по статьям из области информатики.
3. Точность ответов повысилась на 20 %.
4. Модель стала чаще генерировать недостоверные факты.
5. В работе используются адаптеры низкого ранга.

**Exercise 4.** Find a word or phrase in the text with the same meaning.
1. улучшает (точность)
2. уменьшается
3. неверные, ошибочные (факты)
4. первоначальная (модель)
5. область знаний

**Exercise 5.** Fill in на, в, с … до or до.
1. Точность была 85 %, а стала 91 %. Точность выросла ___ 85 % ___ 91 %.
2. Точность выросла ___ 6 %.
3. Процент ошибок был 30 %, а стал 10 %. Процент ошибок уменьшился ___ 10 %.
4. Процент ошибок уменьшился ___ 20 %.
5. Раньше обучение занимало 30 часов, а теперь 10 часов. Время обучения сократилось ___ три раза.

**Exercise 6.** Translate into Russian.
1. Large language models show high results in many tasks.
2. The present paper proposes a fine-tuning method.
3. The approach increases accuracy by 12%.
4. The number of parameters is reduced tenfold.
5. The model generates hallucinations less often.

**Exercise 7.** Write a four-sentence mini-abstract about your own research. Use this template:
В последние годы ___. Однако ___. В настоящей работе предлагается ___. Эксперименты показали, что ___.

## Answer key

**Exercise 1.**
1. Качество работы БЯМ с научными текстами на русском языке остаётся недостаточно изученным.
2. Метод дообучения языковой модели на небольшом корпусе научных статей.
3. На сочетании адаптеров низкого ранга (LoRA) и автоматической генерации пар «вопрос — ответ».
4. 5 000 (пять тысяч) вопросов.
5. На 12 % (на двенадцать процентов).
6. Более чем в сто раз.
7. Адаптация БЯМ к узкой предметной области возможна даже при ограниченных вычислительных ресурсах.

**Exercise 2.** 1. A 2. B 3. C 4. D 5. E

**Exercise 3.** 1. Неверно (даже при ограниченных ресурсах). 2. Верно. 3. Неверно (на 12 %). 4. Неверно (реже). 5. Верно.

**Exercise 4.** 1. повышает 2. сокращается 3. недостоверные 4. исходная 5. предметная область

**Exercise 5.** 1. с, до 2. на 3. до 4. на 5. в

**Exercise 6.**
1. Большие языковые модели показывают высокие результаты во многих задачах.
2. В настоящей работе предлагается метод дообучения.
3. Подход повышает точность на 12 %.
4. Число параметров сокращается в десять раз.
5. Модель реже генерирует галлюцинации.

**Exercise 7.** Model answer (your text will be different):
В последние годы большие языковые модели широко используются для машинного перевода. Однако качество перевода научных текстов остаётся недостаточно изученным. В настоящей работе предлагается метод дообучения модели на корпусе научных статей. Эксперименты показали, что предложенный метод повышает качество перевода на 8 % по сравнению с исходной моделью.

## Review

- You know the moves of an abstract: context, problem, method, data, results, conclusion.
- You can spot the key sentences by their markers: однако, в настоящей работе, эксперименты показали, позволяют сделать вывод.
- You can analyse a long sentence: verb → subject → object → the rest.
- You can read genitive chains and participle phrases such as набор данных, содержащий 5 000 вопросов.
- You can understand and use на 12 %, в сто раз, с 85 % до 91 %, до 10 %.

---
[← Previous: Academic Phrases for Methods and Results (в работе предлагается, было показано)](19-academic-phrases.md) · [Contents](README.md) · [Next: Writing to Your Supervisor: Formal Emails →](21-email-to-professor.md)
