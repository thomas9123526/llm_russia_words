# Lesson 14. Describing Tables, Figures and Results

> **Advanced Russian for Researchers** · Level: C1 · **You will learn:** how to refer to tables and figures (в табл. 2, на рис. 3, как видно из…) · how to compare results: превосходить, уступать, на 1,1 процентного пункта · how to describe trends and dependencies: возрастает, выходит на плато, зависимость … от …

## Text

After the experimental section (Lesson 13), Minho writes section 4 of his article. Ivan Petrovich's advice: «Таблица должна говорить сама, а текст — объяснять, что в ней важно» ("The table should speak for itself, and the text should explain what in it matters"). All numbers are invented for this textbook; the figure itself is not reproduced here, but its values are given in Exercise 4.

**4. Результаты и их обсуждение**

**Таблица 2.** Качество ответов на тестовой выборке RuSciQA, % (среднее ± стандартное отклонение по трём запускам)

| Модель | Метод | EM | F1 | Обучаемые параметры |
|---|---|---|---|---|
| 7 млрд | без дообучения | 41,3 | 58,6 | — |
| 7 млрд | полное дообучение | 63,9 ± 0,7 | 76,2 ± 0,5 | 7 млрд |
| 7 млрд | LoRA (r = 16) | 62,8 ± 0,4 | 75,4 ± 0,3 | 40 млн |
| 13 млрд | без дообучения | 47,5 | 63,1 | — |
| 13 млрд | LoRA (r = 16) | **67,1 ± 0,5** | **79,0 ± 0,4** | 63 млн |

**Примечание.** Без дообучения использовалось жадное декодирование, поэтому разброс между запусками отсутствует. Лучшие результаты выделены полужирным шрифтом.

*Table 2. Answer quality on the RuSciQA test set, % (mean ± standard deviation over three runs). Columns: model, method, EM, F1, trainable parameters. Rows: без дообучения = without fine-tuning; полное дообучение = full fine-tuning. Note: without fine-tuning, greedy decoding was used, so there is no variation between runs. The best results are in bold.*

Результаты экспериментов на тестовой выборке приведены в табл. 2. Как видно из таблицы, дообучение существенно повышает качество ответов для обеих моделей. Так, для модели размером 7 млрд параметров доля точных совпадений возрастает с 41,3 до 62,8 % при использовании LoRA и до 63,9 % при полном дообучении. Таким образом, LoRA уступает полному дообучению всего на 1,1 процентного пункта по EM, хотя число обучаемых параметров при этом сокращается более чем в 150 раз. По метрике F1 разница ещё меньше и составляет 0,8 п. п.; различие между двумя методами статистически незначимо (p > 0,05).

*The results of the experiments on the test set are given in Table 2. As can be seen from the table, fine-tuning substantially improves answer quality for both models. Thus, for the 7-billion-parameter model the exact match rate grows from 41.3 to 62.8% with LoRA and to 63.9% with full fine-tuning. LoRA is therefore only 1.1 percentage points behind full fine-tuning in EM, although the number of trainable parameters is reduced more than 150-fold. In F1 the difference is even smaller, 0.8 p.p.; the difference between the two methods is not statistically significant (p > 0.05).*

Наилучший результат получен для модели размером 13 млрд параметров, дообученной с помощью LoRA: 67,1 % по EM и 79,0 % по F1. Эта модель превосходит меньшую модель с полным дообучением на 3,2 п. п. по EM и на 2,8 п. п. по F1. Стандартные отклонения по трём запускам не превышают 0,7 п. п., что свидетельствует об устойчивости полученных результатов.

*The best result was obtained for the 13-billion-parameter model fine-tuned with LoRA: 67.1% EM and 79.0% F1. This model outperforms the smaller, fully fine-tuned model by 3.2 p.p. in EM and by 2.8 p.p. in F1. The standard deviations over three runs do not exceed 0.7 p.p., which indicates that the results are stable.*

На рис. 3 показана зависимость F1-меры от ранга адаптера r для модели размером 7 млрд параметров. При увеличении ранга с 4 до 16 качество монотонно растёт (с 72,9 до 75,4 %), однако дальнейшее увеличение ранга до 64 не приводит к заметному улучшению: значение F1 выходит на плато и даже незначительно снижается (75,1 % при r = 64). Следовательно, ранг r = 16 можно считать оптимальным компромиссом между качеством ответов и числом обучаемых параметров.

*Figure 3 shows the dependence of F1 on the adapter rank r for the 7-billion-parameter model. As the rank increases from 4 to 16, quality grows monotonically (from 72.9 to 75.4%), but a further increase of the rank to 64 brings no noticeable improvement: F1 reaches a plateau and even decreases slightly (75.1% at r = 64). Hence r = 16 can be considered the optimal trade-off between answer quality and the number of trainable parameters.*

Анализ ошибок показал, что наибольшее число ошибок приходится на числовые вопросы (46 % всех ошибок лучшей модели), тогда как на фактические вопросы приходится лишь 17 %. Это согласуется с выводами ряда работ [12, 15], согласно которым языковые модели испытывают трудности с арифметическими операциями. Остальные 37 % ошибок связаны с открытыми вопросами, на которые модель, как правило, даёт неполный, но не ошибочный ответ; по-видимому, автоматические метрики в этом случае занижают реальное качество ответов.

*Error analysis showed that the largest share of errors falls on numerical questions (46% of all errors of the best model), while factual questions account for only 17%. This agrees with the conclusions of several studies [12, 15], according to which language models have difficulty with arithmetic operations. The remaining 37% of errors concern open questions, to which the model as a rule gives an incomplete but not wrong answer; apparently, the automatic metrics underestimate the real quality of the answers in this case.*

## New words

| Russian | Pronunciation | English |
|---|---|---|
| табли́ца (табл.) | tablítsa | table (ж.) |
| рису́нок (рис.) | risúnok | figure (м.) |
| приведены́ | privedený | are given, are presented (short passive participle, pl.) |
| представле́нный | predstavlénnyy | presented |
| превосходи́ть (+ acc.) | prevoskhodít' | to outperform, to exceed |
| уступа́ть (+ dat.) | ustupát' | to be inferior to, to lag behind |
| проце́нтный пункт (п. п.) | protséntnyy punkt | percentage point (м.) |
| наилу́чший | nailúchshiy | best |
| зави́симость | zavísimost' | dependence (ж.) |
| возраста́ть / возрасти́ | vozrastát' / vozrastí | to grow, to increase |
| снижа́ться / сни́зиться | snizhát'sya / snízit'sya | to decrease, to drop |
| выходи́ть на плато́ | vykhodít' na plató | to reach a plateau |
| моното́нно | monotónno | monotonically |
| незначи́тельно | neznachítel'no | slightly, insignificantly |
| суще́ственно | sushchéstvenno | substantially |
| статисти́чески зна́чимый | statistícheski znáchimyy | statistically significant |
| усто́йчивость | ustóychivost' | stability, robustness (ж.) |
| ана́лиз оши́бок | análiz oshíbok | error analysis (м.) |
| приходи́ться на (+ acc.) | prikhodít'sya na | to account for, to fall on |
| свиде́тельствовать о (+ prep.) | svidétel'stvovat' o | to indicate, to testify to |

## Grammar

### 1. Pointing to tables and figures

Russian distinguishes the prepositions for tables and figures, just as it does for everyday objects: **в таблице** (like в книге, в приложении) but **на рисунке** (like на фотографии, на графике, на диаграмме, на слайде).

| What you want to say | Russian | Case |
|---|---|---|
| The results are given in Table 2. | Результаты приведены в табл. 2. | в + prep. |
| Figure 3 shows the dependence… | На рис. 3 показана зависимость… | на + prep. |
| As can be seen from Table 2, … | Как видно из табл. 2, … | из + gen. |
| It follows from Fig. 3 that … | Из рис. 3 следует, что … | из + gen. |
| According to Table 2, … | Согласно данным табл. 2, … / Согласно табл. 2, … | согласно + dat. |
| (see Table 2) | (см. табл. 2) | — |

**Verbs.** The usual verbs are short passive participles that agree with the subject: результаты **приведены**, зависимость **показана**, примеры **представлены**, значения **даны**. Since the new information is usually *what* is shown, it goes at the end (the theme–rheme rule from Lesson 3): На рис. 3 показана зависимость F1-меры от ранга. If you first introduce the results and then point to where they are, the place goes at the end: Результаты приведены в табл. 2.

**Abbreviations and reading aloud.** In running text write табл. 2 and рис. 3 (with a full stop and a space before the number); in captions write the word Таблица in full. Aloud you expand the abbreviation and read the number after the noun as a plain cardinal in the nominative: в таблице два, на рисунке три — not an ordinal number.

**Captions.** In Russian papers the table title usually stands above the table (Таблица 2. …) and the figure caption below the figure (Рис. 3. …). A useful sentence for the caption note: Лучшие результаты выделены полужирным шрифтом. — *The best results are in bold.*

### 2. Comparing results

Two verbs do most of the work, and both have a fixed set of slots: *whom*, *in what*, *by how much*.

| Verb | Whom | In what | By how much |
|---|---|---|---|
| превосходи́ть | + acc.: превосходит меньшую модель | по + dat.: по F1 | на + acc.: на 2,8 п. п. |
| уступа́ть | + dat.: уступает полному дообучению | по + dat.: по EM | на + acc.: на 1,1 п. п. |
| отстава́ть | от + gen.: отстаёт от лучшей системы | по + dat. | на + acc. |

- Эта модель превосходит меньшую модель на 3,2 п. п. по EM. — *This model outperforms the smaller model by 3.2 p.p. in EM.*
- LoRA уступает полному дообучению всего на 1,1 п. п. — *LoRA is only 1.1 p.p. behind full fine-tuning.*

Other useful patterns: выше / ниже, чем у…; по сравнению с + instr.; разница (различие) между X и Y составляет…; в N раз меньше; различие статистически значимо / незначимо.

**Percent or percentage point?** If accuracy grows from 60 % to 70 %, it grew **на 10 процентных пунктов** (absolute); a growth **на 10 %** would literally mean a relative increase (60 → 66). Russian reviewers notice this. The abbreviation is п. п. Agreement follows the normal number rules:

| Number | Form |
|---|---|
| 1, 21, 31 | на 1 процентный пункт |
| 2–4, 22–24 | на 2 процентных пункта |
| 5–20, 25… | на 5 процентных пунктов |
| any decimal | на 1,1 процентного пункта; на 3,2 процентного пункта |

After a decimal fraction the noun is genitive singular, because it depends on the fraction (одна целая одна десятая *процентного пункта*). This is also why we write 2,5 процента, 0,7 процента.

### 3. Trends and dependencies

To describe a figure, you need verbs of change, adverbs of degree and a few frames.

| Meaning | Verbs | Adverbs |
|---|---|---|
| goes up | растёт, возрастает (возро́с, возросла́, возросло́), увеличивается, повышается | монотонно, резко, плавно |
| goes down | снижается, уменьшается, падает | незначительно, заметно, существенно |
| stays | остаётся практически неизменным, выходит на плато, колеблется в пределах 75–76 % | практически, примерно |
| peaks | достигает максимума при r = 16 | |

Frames:
- **зависимость** X (gen.) **от** Y (gen.): зависимость F1-меры от ранга адаптера.
- **при увеличении / с ростом** + gen.: При увеличении ранга с 4 до 16 качество монотонно растёт. С ростом размера модели прирост снижается.
- **не приводит к** + dat.: Дальнейшее увеличение ранга не приводит к заметному улучшению.
- **чем …, тем …**: Чем больше модель, тем меньше эффект от дообучения.
- **приходиться на** + acc. (a share falls on something): Наибольшее число ошибок приходится на числовые вопросы.
- **свидетельствовать о** + prep.: Это свидетельствует об устойчивости результатов (о → об before a vowel).

Note the past forms of возрасти: возрос, возросла, возросло, возросли — the masculine has no -л-. Точность возросла с 41,3 до 62,8 %. Число ошибок снизилось на треть.

## Study tip

A good results paragraph moves from the general to the specific: (1) where the results are (в табл. 2); (2) the main pattern (дообучение существенно повышает качество); (3) the key numbers that support it (с 41,3 до 62,8 %); (4) the exception or the surprise (LoRA почти не уступает полному дообучению); (5) the interpretation, with a hedge if needed (по-видимому, …). Do not retell the whole table in words: choose two or three comparisons and give the difference, not just the two values.

## Exercises

**Exercise 1.** Answer the questions about the text.
1. Где приведены результаты экспериментов на тестовой выборке?
2. На сколько процентных пунктов LoRA уступает полному дообучению по EM для модели размером 7 млрд параметров?
3. Во сколько раз сокращается число обучаемых параметров при использовании LoRA?
4. Какая модель показала наилучший результат?
5. Что показано на рис. 3?
6. На какие вопросы приходится наибольшее число ошибок?

**Exercise 2.** Fill in в, на or из.
1. Результаты экспериментов приведены ___ табл. 2.
2. ___ рис. 3 показана зависимость F1-меры от ранга адаптера.
3. Как видно ___ табл. 2, дообучение повышает качество ответов.
4. Примеры ошибок модели представлены ___ приложении Б.
5. Как следует ___ рис. 4, качество выходит на плато.
6. ___ графике видно, что F1 монотонно растёт.

**Exercise 3.** Put the words in brackets into the correct form.
1. Модель размером 13 млрд параметров превосходит (меньшая модель) по обеим метрикам.
2. LoRA уступает (полное дообучение) всего на 1,1 п. п.
3. Наша модель отстаёт (от + лучшая система) на 4 п. п.
4. Точность возросла на 3,2 (процентный пункт).
5. Разница составила 2 (процентный пункт).
6. Доля ошибок снизилась на 5 (процентный пункт).
7. На рис. 3 показана зависимость (качество) от (ранг адаптера).

**Exercise 4.** These are the values behind Fig. 3. Верно или неверно?

| r | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| F1, % | 72,9 | 74,3 | 75,4 | 75,3 | 75,1 |

1. При увеличении ранга с 4 до 16 F1-мера монотонно растёт.
2. Максимальное значение F1-меры достигается при r = 64.
3. При r > 16 качество выходит на плато.
4. Разница между r = 4 и r = 16 составляет 2,5 п. п.
5. При r = 8 значение F1-меры выше, чем при r = 32.

**Exercise 5.** Find and correct the mistake in each sentence.
1. Результаты приведены на табл. 2.
2. Как видно в рис. 3, качество выходит на плато.
3. Модель превосходит базовой модели на 5 п. п.
4. LoRA уступает полное дообучение.
5. Точность выросла на 2,5 процентных пунктов.
6. Зависимость качества с ранга показана на рис. 3.

**Exercise 6.** Rewrite in scientific style.
1. Видно по таблице, что большая модель работает лучше.
2. Когда ранг больше 16, лучше уже не становится.
3. LoRA чуть хуже полного дообучения, зато параметров гораздо меньше.
4. Больше всего модель ошибается в вопросах с числами.

**Exercise 7.** Write a results paragraph (5–7 sentences) describing this table.

**Таблица 3.** Точность на тесте RuLogic, % (тест вымышленный)

| Модель | Без дообучения | LoRA |
|---|---|---|
| A (7 млрд) | 52,4 | 68,9 |
| B (13 млрд) | 58,1 | 73,5 |
| C (70 млрд) | 69,7 | 74,2 |

## Answer key

**Exercise 1.**
1. В табл. 2.
2. На 1,1 процентного пункта.
3. Более чем в 150 раз.
4. Модель размером 13 млрд параметров, дообученная с помощью LoRA (67,1 % по EM и 79,0 % по F1).
5. Зависимость F1-меры от ранга адаптера r для модели размером 7 млрд параметров.
6. На числовые вопросы (46 % всех ошибок лучшей модели).

**Exercise 2.** 1. в 2. На 3. из 4. в 5. из 6. На

**Exercise 3.**
1. меньшую модель
2. полному дообучению
3. от лучшей системы
4. процентного пункта
5. процентных пункта
6. процентных пунктов
7. качества, ранга адаптера

**Exercise 4.**
1. Верно.
2. Неверно: максимум достигается при r = 16 (75,4 %).
3. Верно.
4. Верно (75,4 − 72,9 = 2,5).
5. Неверно: при r = 8 F1-мера ниже (74,3 % против 75,3 %).

**Exercise 5.**
1. Результаты приведены в табл. 2.
2. Как видно из рис. 3, качество выходит на плато. (Также верно: На рис. 3 видно, что качество выходит на плато.)
3. Модель превосходит базовую модель на 5 п. п.
4. LoRA уступает полному дообучению.
5. Точность выросла на 2,5 процентного пункта.
6. Зависимость качества от ранга показана на рис. 3.

**Exercise 6.** Model answers (other correct variants are possible):
1. Как видно из табл. 2, модель большего размера показывает более высокие результаты.
2. При ранге выше 16 качество выходит на плато: дальнейшее увеличение ранга не приводит к заметному улучшению.
3. LoRA незначительно уступает полному дообучению, однако число обучаемых параметров при этом сокращается более чем в 150 раз.
4. Наибольшее число ошибок приходится на числовые вопросы.

**Exercise 7.** Model answer:
Результаты на тесте RuLogic приведены в табл. 3. Как видно из таблицы, дообучение с помощью LoRA повышает точность всех трёх моделей. Наибольший прирост наблюдается для модели A: точность возрастает с 52,4 до 68,9 %, то есть на 16,5 п. п. Для модели B прирост составляет 15,4 п. п., а для модели C — всего 4,5 п. п. (с 69,7 до 74,2 %). Наилучший результат получен для модели C с LoRA (74,2 %), однако модель B с LoRA уступает ей лишь на 0,7 п. п., хотя её размер более чем в пять раз меньше. Следовательно, с ростом размера модели эффект от дообучения снижается.

Checklist: (1) the paragraph starts with where the results are (в табл. 3); (2) it states the main pattern before the numbers; (3) differences are given in п. п. with correct agreement (на 16,5 п. п. = на шестнадцать целых пять десятых процентного пункта); (4) comparisons use превосходить + acc. or уступать + dat.; (5) it ends with an interpretation (следовательно, …), not with another number.

## Review

- You can point to tables and figures correctly: в табл. 2, на рис. 3, как видно из…, согласно данным…
- You compare results with превосходить, уступать and отставать, filling the slots *whom / in what / by how much*.
- You distinguish percent from percentage points and agree процентный пункт with any number, including decimals.
- You describe trends and dependencies: возрастает, снижается, выходит на плато, зависимость … от …, при увеличении…, приходится на…
- You can turn a table into a short results paragraph that goes from the general pattern to the key numbers and an interpretation.

---
[← Previous: Describing Experiments, Data and Setup](13-describing-experiments.md) · [Contents](README.md) · [Next: Writing an Abstract and Conclusions →](15-abstract-and-conclusions.md)
