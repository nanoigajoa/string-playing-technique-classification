# Phase 2 Results — Qwen2-Audio Hallucination Evaluation

**Model**: Qwen/Qwen2-Audio-7B-Instruct

**Eval set**: 125 samples × 2 question types = 250 queries

## Overall Accuracy

| Question type | Accuracy |
|---|---|
| Yes/No        | 0.504 |
| Open-ended    | 0.200 |

## Per-class Misclassification Rate

| class             |   n |   yn_miss_rate |   open_miss_rate |   ambiguous_rate |
|:------------------|----:|---------------:|-----------------:|-----------------:|
| ordinario         |  25 |           0.84 |                1 |                0 |
| pizzicato_secco   |  25 |           0.48 |                0 |                0 |
| tremolo           |  25 |           0.36 |                1 |                0 |
| sul_ponticello    |  25 |           0.36 |                1 |                0 |
| col_legno_battuto |  25 |           0.44 |                1 |                0 |

## Phase 1 / Phase 2 Intersection

Phase 1 worst classes : ['sul_ponticello', 'col_legno_battuto']

Phase 2 worst classes : ['ordinario', 'tremolo']

Intersection (Phase 3 case study targets): **[]**

## Notes

- Ambiguous yes/no responses: 0 / 125
- Open-ended wrong technique mentioned alongside correct: 100 / 125
