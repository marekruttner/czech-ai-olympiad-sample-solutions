# AI Robotics – referenční řešení

Toto řešení pokrývá Úkoly 1–5 nad soubory `train.csv` a `test.csv`.

## Co skript dělá

1. **Načte a zkontroluje data**
   - ověří přítomnost `train.csv`, `test.csv`, `robot_id`, `strategy_label` (jen v train),
   - vypíše diagnostiku (tvary, missing values, distribuci tříd).

2. **Spočítá Úkoly 1–4 přesně z `train.csv`**
   - Úkol 1: počet unikátních `arena_type`,
   - Úkol 2: maximum `avg_speed_mps`,
   - Úkol 3: modus `arena_type`,
   - Úkol 4: maximum `items_collected`.

3. **Natrénuje model pro Úkol 5**
   - použije reprodukovatelné seed (`random_state=42`),
   - porovná 3 modely v `StratifiedKFold` s metrikou **Macro-F1**:
     - `LogisticRegression` + OHE,
     - `RandomForestClassifier` + OHE,
     - `HistGradientBoostingClassifier` + ordinal encoding.
   - vybere model s nejlepším průměrným validačním Macro-F1.

4. **Vygeneruje výstupy**
   - `submission.csv` v přesném soutěžním formátu,
   - `metrics.json` s výsledky validace a základní diagnostikou.

## Spuštění

```bash
cd ai-robotics
python solution.py --data-dir . --output-dir .
```

Pokud máte data v jiné složce:

```bash
python solution.py --data-dir /cesta/k/datum --output-dir /cesta/k/vystupu
```

## Poznámky

- Predikce pro Úkol 5 jsou omezené na validní třídy:
  `explorer`, `collector`, `guardian`, `sprinter`.
- Skript provádí validační kontroly formátu `submission.csv` před uložením.


## Notebook varianta

- Interaktivní krok-za-krokem verze je v souboru `reference_solution.ipynb`.
