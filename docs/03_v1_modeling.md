# v1 모델링 — 특징 기반 ML

## 모델 실행 순서

```
Random Forest (베이스라인)
    ↓
XGBoost (주력, 하이퍼파라미터 튜닝)
    ↓
SVM (비교)
```

### 각 모델의 역할

| 모델 | 역할 | 비고 |
|---|---|---|
| Random Forest | 베이스라인 | 튜닝 없이 바로 돌려 기준점 확보 |
| XGBoost | 주력 모델 | Feature Importance / SHAP 해석 대상 |
| SVM | 비교 모델 | 데이터 적을 때 강함, MIR 전통 모델 |

---

## 평가 지표

- **주 지표:** Macro F1 (클래스 불균형 대응)
- **보조:** Confusion Matrix, Per-class Precision / Recall

> 정확도(Accuracy)만 보지 말 것 — 클래스 불균형 시 다수 클래스에 치우쳐 오해를 만든다.

---

## 해석 단계

### Feature Importance (XGBoost 내장)

```python
import xgboost as xgb
import matplotlib.pyplot as plt

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
xgb.plot_importance(model, max_num_features=20)
plt.show()
```

### SHAP Values

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### 해석의 목표

> "X 기법은 Y 특징이 결정적이었고, 그 이유는 음향학적으로 Z이며, 누수는 그룹 분할로 막았다."

특징 나열이 아니라 **인과적 설명 + 검증 절차**를 제시하는 것이 목표다.

---

## 하이퍼파라미터 튜닝 (XGBoost)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
}

gs = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=gkf,
                  scoring='f1_macro', n_jobs=-1)
gs.fit(X, y, groups=recording_id)
```
