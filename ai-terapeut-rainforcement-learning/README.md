# AI Terapeut — Reinforcement Learning solution

## Step 1: Inspection summary

V tomto repozitáři nebyly nalezeny žádné existující soubory pro úlohu `RehabEnv` (ani notebook, ani helpery pro export CSV), takže tato složka obsahuje **samostatné řešení** postavené pro Google Colab / Python.

- `RehabEnv` je implementován v `rehab_solution.py` jako `gymnasium.Env`.
- Prostředí má 5 akcí (`rest`, `passive`, `active`, `strength`, `functional`) a epizodu délky 30 dní.
- Jsou zde 4 profily pacienta: `balanced`, `pain_sensitive`, `low_stamina`, `high_motivation`.
- Chování je mírně stochastické (malý šum v dynamice stavu), ale je řízené seedy.
- Protože v repozitáři není šablona submission CSV, export je definován jako **per-day rollout** se stejnými sloupci pro oba přístupy.

## Rule-based baseline

Pravidlová politika (`rule_based_policy`) používá progresivní, bezpečnou logiku:

1. **Emergency režim**: při vysoké bolesti / únavě volí `rest`.
2. **Early rehab**: při nízké recovery používá `rest`/`passive`.
3. **Mid rehab**: při střední recovery přechází na `active`.
4. **Late rehab**: při vyšší recovery a stabilním stavu používá `strength`.
5. **Advanced rehab**: při velmi dobré připravenosti volí `functional`.

Navíc jsou použity guardy minimální recovery pro každou akci.

## RL algorithm

Použit je **tabulární Q-learning** (zlepšená baseline varianta) nad `gymnasium` prostředím:

- diskretizace stavu: `recovery`, `pain`, `fatigue`, `motivation`, `day_normalized`
- trénink přes všechny 4 profily (profile-balanced sampling)
- epsilon-greedy s decay
- report průběžných metrik každých 500 epizod

## Regenerace výstupů

Spusťte:

```bash
pip install -r ai-terapeut-rainforcement-learning/requirements.txt
python ai-terapeut-rainforcement-learning/rehab_solution.py
```

Skript vytrénuje RL agenta, vyhodnotí obě politiky na stejném setupu a vytvoří:

- `ai-terapeut-rainforcement-learning/rulebased_predictions.csv`
- `ai-terapeut-rainforcement-learning/rl_predictions.csv`

Pro vysvětlení a vizualizace RL procesu je přidaný notebook:

- `ai-terapeut-rainforcement-learning/rehab_explainer.ipynb`

## CSV schema

Oba CSV soubory mají stejný formát (1 řádek = 1 den v epizodě):

- `profile`
- `episode`
- `day`
- `action`
- `action_name`
- `reward`
- `recovery`
- `pain`
- `fatigue`
- `motivation`
