# AI Termika — Referenční řešení

## Přístup

Physics-Informed Neural Network (PINN) pro predikci šíření tepla v kovové tyči.

### Architektura
- MLP: 2 → 128 → 128 → 128 → 128 → 128 → 1
- Aktivace: Tanh (vhodné pro PINN — nekonečněkrát diferencovatelná)
- Inicializace: Xavier normal
- ~66 500 parametrů

### Loss funkce
Kombinace čtyř složek:
- **Data loss** (λ=1.0): MSE na 100 zašuměných trénovacích měřeních
- **Boundary loss** (λ=5.0): MSE na 24 neúplných okrajových bodech
- **Initial loss** (λ=5.0): MSE na 12 řídkých počátečních bodech
- **Physics loss** (λ=0.5): Residual rovnice vedení tepla na 2000 kolokačních bodech

### Trénink
- Optimizer: Adam, lr=1e-3
- LR scheduler: StepLR (snížení 0.5× každých 5000 epoch)
- 15 000 epoch

## Výsledky
- **RMSE ≈ 0.040** na skrytém ground truth

## Soubory
- `solution.ipynb` — kompletní řešení jako Jupyter notebook
- `solution.py` — totéž jako spustitelný Python skript
- `submission.csv` — predikce pro testovací body

## Spuštění
```bash
pip install -r requirements.txt
python solution.py
```

Úroveň řešení: mírně nadprůměrný student (implementuje PINN se všemi složkami loss,
ale nepoužívá adaptivní váhy, learnable alpha, ani residual connections).
