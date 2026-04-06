# Česká AI Olympiáda 2026 – Radar baseline

Tento repozitář obsahuje **minimální baseline řešení** pro úlohu sémantické segmentace radarových heatmap (IOAI 2025 Radar Contest).

## Co script dělá

- stáhne oficiální repozitář IOAI 2025 (sparse checkout jen pro Radar task),
- načte `.mat.pt` tensory `7 x 50 x 181`,
- použije 3vrstvý CNN baseline (`6 -> 16 -> 32 -> 5`),
- trénuje s `CrossEntropyLoss` nad labely posunutými z `[-1, 3]` na `[0, 4]`,
- vyhodnotí IOAI score na tréninku (val/test v oficiálním balíčku jsou bez labelů),
- vygeneruje `submission_val.csv`, `submission_test.csv` a `submission.zip`.

## Instalace

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Spuštění baseline

```bash
python baseline_radar.py \
  --download \
  --data-dir data/ioai-2025 \
  --output-dir outputs \
  --epochs 8 \
  --batch-size 32 \
  --lr 1e-3
```

## Poznámky

- Ve výchozím stavu script očekává složky `training_set`, `validation_set` (nebo `val_set`) a `test_set` (nebo `testing_set`).
- Výstupní predikce v CSV jsou mapované zpět na třídy `-1..3`.
- Trénink je navržen pro Google Colab (GPU), ale poběží i na CPU.
