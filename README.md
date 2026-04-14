# Česká AI Olympiáda 2026 — Vzorová řešení

Repozitář obsahuje vzorová řešení čtyř soutěžních úloh z **České AI Olympiády 2026** ([aiolympiada.cz](https://www.aiolympiada.cz)).

## Struktura repozitáře

| Složka | Úloha | Přístup |
|---|---|---|
| [`ai-perception/`](#ai-perception) | Sémantická segmentace radarových heatmap | CNN (3vrstvý konvoluční model) |
| [`ai-robotics/`](#ai-robotics) | Analýza dat a klasifikace robotů | Scikit-learn (Logistic Regression, Random Forest, HGBC) |
| [`ai-terapeut-rainforcement-learning/`](#ai-terapeut) | Rehabilitační plánování | Tabulární Q-learning + rule-based baseline |
| [`ai-termika/`](#ai-termika) | Predikce šíření tepla v kovové tyči | Physics-Informed Neural Network (PINN) |

---

## AI Perception

**Úloha:** Sémantická segmentace radarových heatmap z datasetu IOAI 2025 Radar Contest.

- Vstup: `.mat.pt` tensory `7 × 50 × 181`
- Model: 3vrstvý CNN (`6 → 16 → 32 → 5 tříd`)
- Trénink: `CrossEntropyLoss`, labely posunuté z `[-1, 3]` na `[0, 4]`
- Výstup: `submission_val.csv`, `submission_test.csv`, `submission.zip`

```bash
cd ai-perception
pip install -r requirements.txt
python baseline_radar.py --download --data-dir data/ioai-2025 --output-dir outputs --epochs 8
```

**Soubory:** `baseline_radar.py`, `baseline_radar_learning.ipynb`

---

## AI Robotics

**Úloha:** Analýza dat robotů (Úkoly 1–4) a klasifikace strategií robotů (Úkol 5).

- Úkoly 1–4: statistické výpočty z `train.csv` (unikátní arény, max rychlost, modus, max sebraných předmětů)
- Úkol 5: klasifikace `strategy_label` pomocí `StratifiedKFold` cross-validace s metrikou Macro-F1
- Modely: `LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`
- Výstup: `submission.csv`, `metrics.json`

```bash
cd ai-robotics
pip install -r requirements.txt
python solution.py --data-dir . --output-dir .
```

**Soubory:** `solution.py`, `reference_solution.ipynb`, `README_solution.md`

---

## AI Terapeut

**Úloha:** Rehabilitační plánování pomocí reinforcement learningu v prostředí `RehabEnv` (Gymnasium).

- Prostředí: 5 akcí (`rest`, `passive`, `active`, `strength`, `functional`), epizoda = 30 dní
- 4 profily pacienta: `balanced`, `pain_sensitive`, `low_stamina`, `high_motivation`
- Rule-based baseline: progresivní politika s bezpečnostními pravidly
- RL agent: tabulární Q-learning s epsilon-greedy a decay
- Výstup: `rulebased_predictions.csv`, `rl_predictions.csv`

```bash
cd ai-terapeut-rainforcement-learning
pip install -r requirements.txt
python rehab_solution.py
```

**Soubory:** `rehab_solution.py`, `rehab_explainer.ipynb`

---

## AI Termika

**Úloha:** Predikce šíření tepla v kovové tyči pomocí Physics-Informed Neural Network.

- Vstup: zašuměná měření, neúplné okrajové a počáteční podmínky, přibližná difuzivita α
- Model: MLP `2 → 128 → 128 → 128 → 128 → 128 → 1` s Tanh aktivací (~66 500 parametrů)
- Loss: data (λ=1.0) + boundary (λ=5.0) + initial (λ=5.0) + physics residual (λ=0.5)
- Trénink: Adam, lr=1e-3, StepLR scheduler, 15 000 epoch
- Výsledek: RMSE ≈ 0.040
- Výstup: `submission.csv`

```bash
cd ai-termika
pip install -r requirements.txt
python solution.py
```

**Soubory:** `solution.py`, `solution.ipynb`, `baseline.py`, `baseline.ipynb`, `ai_termika.md` (zadání)

---

## Rychlý start

```bash
git clone <repo-url>
cd czech-ai-olympiad-sample-solutions
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Každá složka má vlastní `requirements.txt` s konkrétními závislostmi pro danou úlohu.

## Licence

Zadání úloh: CC BY-NC-SA 4.0 | Organizátor: [nvias, z.s.](https://www.nvias.org)
