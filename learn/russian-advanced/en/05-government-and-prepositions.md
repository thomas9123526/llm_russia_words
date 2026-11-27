# Lesson 5. Government: Which Case After Which Word (влиять на, зависеть от, благодаря, согласно)

> **Advanced Russian for Researchers** · Level: B2 · **You will learn:** the government of key research verbs, nouns and adjectives (влиять на + acc., зависеть от + gen., способствовать + dat.) · secondary prepositions of cause, basis and comparison (благодаря, согласно, вследствие, по сравнению с, по окончании) · how government works in verbal nouns and before a что-clause (о том, что; на то, что)

## Text

Minho is writing the "Discussion" section of his article. Here is the version he sent to Ivan Petrovich after careful checking of every preposition and case. (The text was written for this textbook; the numbers and the cited work [7] are examples.)

**Обсуждение результатов**

> Полученные результаты свидетельствуют о том, что размер словаря существенно влияет на качество модели. Согласно данным таблицы 3 средняя длина последовательности при увеличении словаря с 32 до 64 тысяч токенов сокращается почти на 20 %. Благодаря этому модель обрабатывает больше текста за то же время.
>
> Вместе с тем качество зависит не только от размера словаря, но и от состава обучающего корпуса. В отличие от новостных текстов, научные статьи содержат много терминов и формул. Из-за этого токенизатор, обученный на новостях, разбивает такие термины на слишком мелкие фрагменты. По сравнению с базовой моделью наша модель лучше справляется с научными текстами, однако на коротких диалогах она уступает ей по точности.
>
> Следует обратить внимание на ещё один результат. Независимо от размера модели прирост качества оказался примерно одинаковым. Это указывает на то, что эффект связан в первую очередь с токенизацией, а не с числом параметров. Данный вывод соответствует результатам, полученным в работе [7], и не противоречит более ранним исследованиям.
>
> Отдельно следует сказать о выборе метрик. Мы опирались на метрики, которые широко используются в предшествующих работах, и сравнивали все модели с одной и той же базовой моделью. Такой подход способствует воспроизводимости результатов и позволяет избежать случайных выводов.
>
> Вследствие ограниченных вычислительных ресурсов мы не смогли проверить модели с числом параметров более 7 млрд. Тем не менее мы рассчитываем на то, что выявленная закономерность сохранится и для более крупных моделей. В ходе дальнейшей работы мы планируем уделить особое внимание многоязычным корпусам.
>
> По окончании экспериментов все модели и код будут опубликованы в открытом доступе. Авторы благодарят коллег по кафедре за помощь в разметке данных и рецензентов за ценные замечания.

*Translation:*

*Discussion of the results. The results obtained indicate that the vocabulary size has a substantial effect on model quality. According to the data in Table 3, when the vocabulary is increased from 32 to 64 thousand tokens, the average sequence length decreases by almost 20%. Thanks to this, the model processes more text in the same time.*

*At the same time, quality depends not only on the vocabulary size but also on the composition of the training corpus. Unlike news texts, scientific articles contain many terms and formulas. Because of this, a tokenizer trained on news splits such terms into fragments that are too small. Compared with the baseline model, our model copes better with scientific texts; however, on short dialogues it is inferior to it in accuracy.*

*One more result deserves attention. Regardless of model size, the gain in quality turned out to be roughly the same. This indicates that the effect is connected primarily with tokenization rather than with the number of parameters. This conclusion is consistent with the results obtained in [7] and does not contradict earlier studies.*

*A separate word should be said about the choice of metrics. We relied on metrics that are widely used in previous work and compared all models with one and the same baseline model. This approach contributes to the reproducibility of the results and makes it possible to avoid accidental conclusions.*

*Owing to limited computational resources, we could not test models with more than 7 billion parameters. Nevertheless, we expect that the regularity we found will also hold for larger models. In the course of further work, we plan to pay special attention to multilingual corpora.*

*Upon completion of the experiments, all models and the code will be made openly available. The authors thank their colleagues at the department for help with data annotation and the reviewers for valuable comments.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| управле́ние | upravlénie | government (in grammar) (ср.) |
| влия́ть / повлия́ть на + acc. | vliyát' / povliyát' na | to influence, to affect |
| влия́ние на + acc. | vliyánie na | influence on, effect on (ср.) |
| зави́сеть от + gen. | zavíset' ot | to depend on |
| свиде́тельствовать о + prep. | svidétel'stvovat' o | to indicate, to testify to |
| благодаря́ + dat. | blagodaryá | thanks to |
| согла́сно + dat. | soglásno | according to |
| всле́дствие + gen. | vslédstvie | owing to, as a consequence of (formal) |
| в отли́чие от + gen. | v otlíchie ot | unlike |
| по сравне́нию с + instr. | po sravnéniyu s | compared with |
| справля́ться / спра́виться с + instr. | spravlyát'sya / správit'sya s | to cope with |
| уступа́ть / уступи́ть + dat. | ustupát' / ustupít' | to be inferior to |
| соотве́тствовать + dat. | sootvétstvovat' | to correspond to, to be consistent with |
| противоре́чить + dat. | protivoréchit' | to contradict |
| опира́ться / опере́ться на + acc. | opirát'sya / operét'sya na | to rely on, to build on |
| спосо́бствовать + dat. | sposóbstvovat' | to contribute to, to promote |
| избега́ть / избежа́ть + gen. | izbegát' / izbezhát' | to avoid |
| рассчи́тывать на + acc. | rasschítyvat' na | to count on, to expect (сч is pronounced [щ]) |
| уделя́ть / удели́ть внима́ние + dat. | udelyát' / udelít' vnimánie | to pay attention to |
| по оконча́нии + gen. | po okonchánii | upon completion of |
| закономе́рность | zakonomérnost' | regularity, pattern (ж.) |
| воспроизводи́мость | vosproizvodímost' | reproducibility (ж.) |
| разме́тка | razmétka | annotation, labelling (of data) (ж.) |

## Grammar

**Government** (управление) is the rule that a word requires a particular case, with or without a particular preposition: влиять **на** что, зависеть **от** чего, способствовать **чему**. For a Korean speaker this is close to choosing a particle (조사): *~에 영향을 미치다*, *~에 따라 달라지다*. The trouble is that neither Korean nor English predicts the Russian choice: *depend on* is зависеть **от** (literally *from*), *pay attention to* is уделять внимание + dative **without** a preposition. Government must be learned together with the word, as one unit.

Russians themselves quote government with question words: влиять на **что**, зависеть от **чего**, соответствовать **чему**, пользоваться **чем**, свидетельствовать о **чём**. Learn new verbs in this form.

### 1. Key research words by construction

| Construction | Verbs, nouns, adjectives | Example |
|---|---|---|
| **на** + acc. | влиять / влияние на, указывать на, обращать внимание на, опираться на, рассчитывать на, направлен на, ссылаться / ссылка на, рецензия на, разбивать на | Метод направлен на сокращение вычислений. |
| **на** + prep. | основываться на, основан на, сосредоточиться на, обучать на (данных) | Метод основан на механизме внимания. |
| **о** + prep. | свидетельствовать о, сообщать о, говорить о | Данные свидетельствуют о высокой точности метода. |
| **от** + gen. | зависеть от, зависимость от, отличаться от, отказаться от, независимо от | Наш метод отличается от базового числом параметров. |
| **с** + instr. | сравнивать с, связан с, справляться с, совпадать с, согласен с | Ошибка связана с качеством разметки. |
| **к** + dat. | интерес к, применять к, приводить к, относиться к, подход к, стремиться к | Это приводит к потере информации. |
| **в** + prep. | участвовать в, нуждаться в, сомневаться в, уверенность в, заключаться в | Мы нуждаемся в дополнительных данных. |
| dat., no preposition | способствовать, препятствовать, соответствовать, противоречить, уступать, уделять внимание, посвящён | Работа посвящена проблеме токенизации. |
| instr., no preposition | пользоваться, руководить, заниматься, обладать, являться, характеризоваться | Мы пользовались открытой библиотекой. |
| gen., no preposition | достигать / достичь, избегать / избежать, касаться | Модель достигла точности 92 %. |
| acc., no preposition | использовать, исследовать, анализировать, учитывать, обсуждать, превосходить | Наша модель превосходит базовую по скорости. |

Some pairs differ by one detail and are often confused:

- **обращать внимание на** + acc. but **уделять внимание** + dat.: обратить внимание на качество данных / уделить внимание качеству данных.
- **пользоваться** + instr. but **использовать** + acc.: пользоваться библиотекой / использовать библиотеку.
- **превосходить** + acc. **по** + dat. and **уступать** + dat. **по** + dat.: Модель А превосходит модель Б по скорости, но уступает ей по точности.
- **уверенность в** + prep. but **вера в** + acc.: уверенность в успехе / вера в успех.
- **оплатить** + acc. but **заплатить за** + acc.: оплатить доступ к кластеру / заплатить за доступ к кластеру.
- **рецензия на** + acc.: рецензия на статью. With **отзыв** both are used: отзыв о работе, отзыв на автореферат.

Remember that after a preposition the third-person pronouns take **н-**: влиять на **него**, зависеть от **неё**, сравнивать с **ними**. Without a preposition there is no н-: соответствовать **им**, пользоваться **ею**.

### 2. Secondary prepositions: cause, basis, time, comparison

Scientific Russian uses many prepositions formed from nouns, adverbs and verbal adverbs. Each has a fixed case.

| Preposition | Case | Meaning and use |
|---|---|---|
| **благодаря** | dat. | *thanks to*: a cause with a good or neutral result. Благодаря этому модель обрабатывает больше текста. |
| **из-за** | gen. | *because of*: usually a cause with a bad result. Из-за сбоя сервера мы потеряли два дня. |
| **вследствие** | gen. | *owing to*: formal, neutral cause. Вследствие ограниченных ресурсов… |
| **в результате** | gen. | *as a result of*. В результате увеличения словаря… |
| **согласно** | dat. | *according to* (data, rules, plans). Согласно данным таблицы 3… |
| **вопреки** | dat. | *contrary to*. Вопреки ожиданиям качество не снизилось. |
| **в соответствии с** | instr. | *in accordance with*. В соответствии с требованиями журнала… |
| **на основе / на основании** | gen. | *on the basis of*. на основе открытых данных |
| **с помощью / при помощи** | gen. | *by means of, with the help of*. с помощью механизма внимания |
| **в течение** | gen. | *during, for* (a period). в течение всего семестра |
| **в ходе / в процессе** | gen. | *in the course of*. в ходе дальнейшей работы |
| **с целью / в целях** | gen. | *with the aim of*. с целью проверки гипотезы |
| **в отличие от / независимо от** | gen. | *unlike / regardless of*. |
| **по сравнению с / в связи с** | instr. | *compared with / in connection with*. |
| **по отношению к** | dat. | *in relation to*. |
| **по** (= *after*) | prep. | по окончании, по завершении, по прибытии, по приезде |

Three points need special care.

**Благодаря** is not a neutral *because of*. It keeps a trace of *thanks*, so ~~благодаря ошибке в коде модель обучалась неправильно~~ sounds absurd (or ironic). For a negative cause use **из-за**, in formal writing **вследствие**.

**Согласно** takes the **dative**: согласно данным, согласно плану, согласно правилам. The genitive (~~согласно плана~~) is a widespread error, also among native speakers, and reviewers notice it.

**По** meaning *after, upon* takes the **prepositional** and ends in **-и** or **-е**: по окончании экспериментов, по завершении проекта, по приезде в Москву. Compare ordinary **по** + dative: по данным, по точности.

Note the spelling: **в течение** (time) ends in **-е**, and **вследствие** is one word. Phrases with благодаря, согласно, вследствие, в отличие от, независимо от can be set off by commas, especially when they are long or stand at the beginning, but this is optional. In this lesson's exercises we do not add such commas.

### 3. Government in verbal nouns and before a clause

**Verbal nouns keep the preposition of their verb:**

| Verb | Noun |
|---|---|
| влиять на качество | влияние на качество |
| зависеть от данных | зависимость от данных |
| интересоваться лингвистикой | интерес к лингвистике (the noun takes к + dat.) |
| руководить проектом | руководство проектом |

**A transitive verb's accusative object becomes a genitive:** анализировать результаты → анализ результатов; использовать токенизатор → использование токенизатора. The doer, if mentioned, goes into the instrumental: анализ ошибок авторами. When a noun has both the subject of the verb and a prepositional object, the order is noun + genitive + prepositional phrase: размер словаря влияет на качество модели → **влияние размера словаря на качество модели**.

**Before a что-clause**, a preposition cannot stand directly before **что**. Russian inserts the pronoun **то** in the case the preposition requires:

| Government | Before a clause |
|---|---|
| свидетельствовать о + prep. | свидетельствует **о том, что**… |
| указывать на + acc. | указывает **на то, что**… |
| приводить к + dat. | приводит **к тому, что**… |
| связан с + instr. | связано **с тем, что**… |
| зависеть от + gen. | зависит **от того**, какой… / как… |
| заключаться в + prep. | заключается **в том, что**… |

Verbs with a plain accusative do not need **то**: показывать, доказывать, утверждать, считать — Мы показали, что…; Автор доказывает, что… (~~доказывает о том, что~~ is wrong).

**Homogeneous verbs with different government cannot share one object.** ~~Мы изучили и опираемся на работы предшественников~~ is wrong, because изучить takes the accusative without a preposition and опираться takes на + acc. Give each verb its own object: Мы изучили работы предшественников и опираемся на них.

## Study tip

**Keep a government notebook.** Every time you meet a useful verb in a Russian paper, write it down with its "slot": соответствовать **чему**, превосходить **что по чему**, рассчитывать **на что**. Add one real sentence from the paper. For checking, use a dictionary of government (словарь управления) or the examples in the National Corpus of the Russian Language: search for the verb and look at what follows it. When you write in Korean, the particle comes after the noun; in Russian the preposition comes before it and the case ending after it, so check both ends of the phrase.

## Exercises

**Exercise 1.** Put the words in brackets into the correct case.
1. Размер словаря влияет на (качество модели).
2. Скорость обучения зависит от (объём данных).
3. Согласно (результаты эксперимента) метод работает быстрее.
4. Благодаря (новый токенизатор) качество выросло.
5. Из-за (ошибка в данных) обучение пришлось повторить.
6. Мы уделили особое внимание (качество разметки).
7. Полученные данные свидетельствуют о (высокая точность метода).
8. Этот вывод противоречит (результаты работы [3]).
9. Наш метод основан на (механизм внимания).
10. Мы пользовались (открытая библиотека).
11. Модель достигла (точность 92 %).
12. В течение (весь семестр) мы собирали данные.

**Exercise 2.** Fill in the preposition. If no preposition is needed, write —.
1. Метод направлен … сокращение вычислений.
2. Работа посвящена … проблеме токенизации.
3. Мы сравнили нашу модель … базовой.
4. Такой подход способствует … ускорению обучения.
5. Это приводит … потере информации.
6. Модель хорошо справляется … этой задачей.
7. Мы нуждаемся … дополнительных данных.
8. Метод применяли … задаче классификации.
9. Проект соответствует … требованиям фонда.
10. … окончании эксперимента мы сохранили все модели.
11. Наш метод отличается … базового числом параметров.
12. Мы опирались … результаты предыдущих работ.

**Exercise 3.** Turn the verb phrases into noun phrases.
1. влиять на качество
2. зависеть от данных
3. анализировать результаты
4. использовать токенизатор
5. сравнивать модели
6. интересоваться лингвистикой
7. руководить проектом
8. исследовать токенизацию
9. Размер словаря влияет на качество модели.
10. Авторы анализируют ошибки.

**Exercise 4.** Fill in the gap with the correct form of то and its preposition (о том, на то, к тому, с тем, от того, в том).
1. Результаты свидетельствуют …, что метод эффективен.
2. Это указывает …, что модель переобучилась.
3. Увеличение словаря приводит …, что последовательности становятся короче.
4. Ошибка связана …, что данные не были очищены.
5. Качество зависит …, какой токенизатор используется.
6. Мы рассчитываем …, что результаты подтвердятся.
7. Проблема заключается …, что данных мало.

**Exercise 5.** Correct the mistake in each sentence.
1. Согласно плана защита состоится весной.
2. Нужно уделить внимание на качество данных.
3. Мы оплатили за доступ к кластеру.
4. Результаты свидетельствуют, что метод работает.
5. По приезду в Москву я сразу пошёл в деканат.
6. Наш метод отличается с базового.
7. Благодаря ошибке в коде модель обучалась неправильно.
8. Рецензент выразил уверенность в успех проекта.
9. Мы изучили и опираемся на работы предшественников.
10. Автор доказывает о том, что метод эффективен.
11. Рецензия о статье пришла вчера.

**Exercise 6.** Rewrite in scientific style, using the word in brackets.
1. Результат стал лучше, потому что мы взяли новый токенизатор. (благодаря)
2. Как написано в таблице 2, модель работает быстрее. (согласно)
3. Модель может быть большой или маленькой, но качество растёт одинаково. (независимо от)
4. Мы не проверили большие модели, потому что у нас мало ресурсов. (вследствие)
5. Когда эксперименты закончатся, мы опубликуем код. (по окончании)
6. Наша модель работает быстрее, чем базовая. (по сравнению с)

**Exercise 7.** Translate into Russian.
1. The quality of the model depends on the size of the corpus.
2. We studied the influence of the vocabulary size on the speed of generation.
3. This result corresponds to the data of previous studies.
4. Thanks to my supervisor, I finished the paper on time.
5. The paper is devoted to the problem of tokenization.
6. Because of a server failure, we lost two days.

**Exercise 8.** Write a paragraph (4–6 sentences) about the factors that affect the results of your own model. Use at least five of these: влиять на, зависеть от, благодаря, из-за, по сравнению с, соответствовать, способствовать, согласно.

## Answer key

**Exercise 1.**
1. на качество модели
2. от объёма данных
3. Согласно результатам эксперимента
4. Благодаря новому токенизатору
5. Из-за ошибки в данных
6. качеству разметки
7. о высокой точности метода
8. результатам работы [3]
9. на механизме внимания
10. открытой библиотекой
11. точности 92 %
12. В течение всего семестра

**Exercise 2.**
1. на
2. —
3. с
4. —
5. к
6. с
7. в
8. к
9. —
10. По
11. от
12. на

**Exercise 3.**
1. влияние на качество
2. зависимость от данных
3. анализ результатов
4. использование токенизатора
5. сравнение моделей
6. интерес к лингвистике
7. руководство проектом
8. исследование токенизации
9. влияние размера словаря на качество модели
10. анализ ошибок авторами

**Exercise 4.**
1. о том
2. на то
3. к тому
4. с тем
5. от того
6. на то
7. в том

**Exercise 5.**
1. Согласно плану защита состоится весной.
2. Нужно уделить внимание качеству данных. *also:* Нужно обратить внимание на качество данных.
3. Мы оплатили доступ к кластеру. *also:* Мы заплатили за доступ к кластеру.
4. Результаты свидетельствуют о том, что метод работает.
5. По приезде в Москву я сразу пошёл в деканат.
6. Наш метод отличается от базового.
7. Из-за ошибки в коде модель обучалась неправильно.
8. Рецензент выразил уверенность в успехе проекта.
9. Мы изучили работы предшественников и опираемся на них.
10. Автор доказывает, что метод эффективен.
11. Рецензия на статью пришла вчера.

**Exercise 6.**
1. Благодаря новому токенизатору результат улучшился. *also:* Результат улучшился благодаря новому токенизатору.
2. Согласно данным таблицы 2 модель работает быстрее. *also:* Согласно таблице 2 модель работает быстрее.
3. Независимо от размера модели качество растёт одинаково.
4. Вследствие ограниченных ресурсов мы не проверили большие модели. *also:* Вследствие нехватки ресурсов мы не проверили большие модели.
5. По окончании экспериментов мы опубликуем код.
6. По сравнению с базовой моделью наша модель работает быстрее.

**Exercise 7.**
1. Качество модели зависит от размера корпуса.
2. Мы исследовали влияние размера словаря на скорость генерации. *also:* Мы изучили влияние размера словаря на скорость генерации.
3. Этот результат соответствует данным предыдущих исследований.
4. Благодаря научному руководителю я закончил статью вовремя. *also:* Благодаря моему научному руководителю я вовремя закончил статью.
5. Статья посвящена проблеме токенизации.
6. Из-за сбоя сервера мы потеряли два дня.

**Exercise 8.** *Model answer:*
Качество нашей модели зависит от нескольких факторов. В первую очередь на него влияет размер обучающего корпуса. Благодаря дополнительным данным точность выросла на 4 %. Однако из-за нехватки памяти мы не смогли увеличить размер модели. По сравнению с базовой моделью наша модель работает быстрее, что соответствует результатам предыдущих работ.

Checklist:
- влиять на + acc., зависеть от + gen., соответствовать + dat., способствовать + dat. (no preposition);
- благодаря + dat. only for a good or neutral cause, из-за + gen. for a bad one;
- согласно + dat. (согласно данным, not ~~согласно данных~~);
- pronouns after prepositions with н-: на него, от неё;
- before a что-clause: о том, что / на то, что / к тому, что.

## Review

- You learn verbs together with their government: влиять на что, зависеть от чего, соответствовать чему, пользоваться чем.
- You choose between благодаря + dat. and из-за + gen. by the meaning of the result, and use вследствие + gen. in formal writing.
- You write согласно данным, по окончании, в течение семестра without the common mistakes.
- You form noun phrases with the right government: влияние размера словаря на качество модели, анализ результатов.
- You insert то before a что-clause: свидетельствует о том, что…; указывает на то, что…; and you do not join verbs with different government to one object.

---
[← Previous: Punctuation: Commas in Complex Sentences, Introductory Words, Dashes](04-punctuation.md) · [Contents](README.md) · [Next: The Scientific Register: Formal, Neutral and Colloquial Russian →](06-scientific-register.md)
