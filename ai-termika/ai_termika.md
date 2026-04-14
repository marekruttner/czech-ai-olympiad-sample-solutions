# ČESKÁ AI OLYMPIÁDA 2026

**www.aiolympiada.cz**

---

## ZADÁNÍ — AI Tech (individuální soutěž)

### AI Termika — Učení fyziky z dat pomocí neuronových sítí

| | |
|---|---|
| **Formát** | Jednotlivec \| Samostatná práce, bez konzultací |
| **Čas** | 4 hodiny (krajské kolo) |
| **Prostředí** | Google Colab (Python 3, GPU runtime) |
| **Hodnocení** | Automatizované — RMSE na skrytých testovacích bodech (100 %) |
| **Odevzdání** | `submission.csv` + zdrojový kód v notebooku |
| **Organizátor** | nvias, z.s. |
| **Licence** | CC BY-NC-SA 4.0 |

---

## 1. Úvod

V mnoha reálných situacích máme jen několik měření, ale známe fyzikální zákon, který systém přibližně popisuje. Klasická neuronová síť se učí jen z dvojic vstup–výstup. Existuje ale přístup zvaný **Physics-Informed Neural Network (PINN)**, který do ztrátové funkce sítě přidává i fyzikální rovnici — model se tak učí zároveň z dat i z fyziky.

V této úloze budete předpovídat šíření tepla v kovové tyči. K dispozici máte omezený počet zašuměných měření, neúplné okrajové podmínky a přibližnou fyzikální rovnici. Vaším cílem je vytvořit neuronovou síť, která co nejpřesněji odhadne teplotu v zadaných testovacích bodech.

---

## 2. Popis problému

### 2.1 Fyzikální pozadí

Máme tenkou kovovou tyč. V průběhu času se v ní šíří teplo. Teplota v tyči je popsána funkcí $u(x, t)$, kde $x$ je poloha a $t$ je čas. Všechny veličiny jsou v **bezrozměrných normalizovaných jednotkách**:

- $x \in [0, 1]$
- $t \in [0, 1]$
- $u \in \mathbb{R}$ (bezrozměrná teplota)

Šíření tepla přibližně popisuje 1D rovnice vedení tepla:

$$\frac{\partial u}{\partial t} = \alpha \cdot \frac{\partial^2 u}{\partial x^2}$$

kde $\alpha$ je bezrozměrná konstanta difuzivity.

### 2.2 Co je v úloze jiné oproti učebnici

V reálném inženýrství nemáme nikdy dokonalé informace. Proto i v této úloze platí:

- měření teploty obsahují **šum**,
- okrajové podmínky jsou známé jen **částečně** (ne pro všechny časy),
- počáteční podmínka je zadána jen v **několika bodech** (ne spojitě),
- hodnota $\alpha$ je **přibližná** — skutečný systém se může mírně odchylovat.

Díky tomu nelze úlohu spolehlivě vyřešit pouhým numerickým řešením diferenciální rovnice. Úspěšné řešení vyžaduje model, který umí zkombinovat neúplné fyzikální znalosti s reálnými daty.

---

## 3. Úloha

> **🎯 Cíl**
>
> Navrhněte neuronovou síť, která pro zadané body $(x, t)$ odhadne teplotu $u(x, t)$. Využijte dostupná měření, fyzikální rovnici i neúplné okrajové a počáteční podmínky. Výsledný model musí generovat predikce ve formátu `submission.csv`.

> **⚠️ Pravidlo:** Řešení musí být postaveno na neuronové síti (MLP, PINN nebo jiná architektura). Čistě numerická řešení PDE (finite differences, finite elements apod.) nejsou přípustná jako hlavní model. Fyzikální rovnici ale můžete (a měli byste) využít jako součást tréninku sítě.

---

## 4. Poskytnuté soubory

| Soubor | Popis | Sloupce |
|---|---|---|
| `train_measurements.csv` | Zašuměná měření teploty v rozptýlených bodech uvnitř domény (~50–100 bodů) | `x`, `t`, `u` |
| `boundary_partial.csv` | Známé hodnoty teploty na krajích tyče ($x = 0$ a $x = 1$), ale **jen pro některé časy** (~20–30 bodů) | `x`, `t`, `u` |
| `initial_sparse.csv` | Teplota v čase $t = 0$, ale jen v **několika bodech** (~10–15 bodů) | `x`, `t`, `u` |
| `test_points.csv` | Body, pro které máte predikovat teplotu (~500 bodů) | `id`, `x`, `t` |
| `constants.json` | Přibližná hodnota $\alpha$ a rozsah domény | — |

> **📌 Poznámky k datům:**
>
> - Měření v `train_measurements.csv` obsahují aditivní Gaussovský šum.
> - Soubory `boundary_partial.csv` a `initial_sparse.csv` jsou neúplné — nepokrývají celou hranici ani celý počáteční stav.
> - Hodnota $\alpha$ v `constants.json` je přibližná (±5–10 % od skutečné hodnoty).
> - Testovací body leží uvnitř domény $[0,1] \times [0,1]$, žádné body nejsou extrapolační.
> - Data nemají duplicity a neobsahují chybějící hodnoty (NaN).

Všechny soubory budou dostupné ke stažení na platformě **platform.aiolympiada.cz** a v přiloženém baseline notebooku.

---

## 5. Doporučený přístup

Úloha je navržena tak, aby šla řešit více způsoby s rostoucí sofistikovaností.

### 5.1 Základní řešení — data-only MLP

Obyčejná MLP síť trénovaná pouze na měřených datech. Model se učí mapování $(x, t) \to u$ výhradně z pozorování. Toto řešení je funkční, ale na řídkých zašuměných datech bude mít omezenou přesnost.

### 5.2 Silnější řešení — PINN

Physics-Informed Neural Network, kde ztrátová funkce obsahuje více složek:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_1 \cdot \mathcal{L}_{\text{boundary}} + \lambda_2 \cdot \mathcal{L}_{\text{initial}} + \lambda_3 \cdot \mathcal{L}_{\text{physics}}$$

kde:

- $\mathcal{L}_{\text{data}}$ — chyba na trénovacích měřeních,
- $\mathcal{L}_{\text{boundary}}$ — chyba na známých okrajových bodech,
- $\mathcal{L}_{\text{initial}}$ — chyba na známých počátečních bodech,
- $\mathcal{L}_{\text{physics}}$ — residual rovnice vedení tepla na kolokačních bodech:

$$\mathcal{L}_{\text{physics}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \left( \frac{\partial u}{\partial t}\bigg|_{(x_i, t_i)} - \alpha \cdot \frac{\partial^2 u}{\partial x^2}\bigg|_{(x_i, t_i)} \right)^2$$

Kolokační body jsou náhodně vzorkované body v doméně, kde model nemá žádná měření. Parciální derivace modelu se počítají pomocí **automatické diferenciace** (`torch.autograd.grad`).

### 5.3 Tipy pro vylepšení

- Adaptivní váhy $\lambda_1, \lambda_2, \lambda_3$
- Hlubší nebo širší MLP (4–6 vrstev, 64–256 neuronů)
- Aktivační funkce `Tanh` (pro PINN výhodnější než `ReLU`)
- Learning rate scheduling
- Residual connections

---

## 6. Baseline model

Přiložený notebook obsahuje jednoduchou MLP architekturu jako výchozí bod:

```
Vstup: 2 neurony (x, t)
Linear(2 → 64) + Tanh
Linear(64 → 64) + Tanh
Linear(64 → 64) + Tanh
Linear(64 → 1) → Výstup: predikovaná teplota u(x, t)
```

Notebook obsahuje:

- načtení všech CSV souborů,
- definici MLP sítě,
- trénovací smyčku (pouze data loss),
- skeleton PINN loss (k doplnění),
- export `submission.csv`.

> **💡 Tip:** Baseline model používá jen $\mathcal{L}_{\text{data}}$. Vaším úkolem je přidat fyzikální a okrajové složky loss funkce a optimalizovat celý přístup.

---

## 7. Hodnocení

### 7.1 Hlavní skóre (100 %)

Hodnocení je **plně automatizované**. Na neveřejných testovacích bodech (skrytý ground truth) se spočítá RMSE mezi predikovanou a skutečnou teplotou:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( u_{\text{pred}}(x_i, t_i) - u_{\text{true}}(x_i, t_i) \right)^2}$$

**Ranking se určuje přímo podle RMSE — čím nižší, tím lepší.** Žádná dodatečná normalizace.

### 7.2 Tie-break: physics residual

V případě shodného RMSE (zaokrouhleného na 4 desetinná místa) rozhoduje průměrný absolutní physics residual spočítaný organizátorem na sadě kontrolních bodů:

$$R = \frac{1}{M} \sum_{j=1}^{M} \left| \frac{\partial u}{\partial t}\bigg|_{(x_j, t_j)} - \alpha_{\text{true}} \cdot \frac{\partial^2 u}{\partial x^2}\bigg|_{(x_j, t_j)} \right|$$

Nižší residual = lepší umístění. Tento výpočet provádí organizátor z odevzdaného notebooku po skončení soutěže a **slouží výhradně jako tie-break**, nikoli jako součást hlavního rankingu.

### 7.3 Bodování

Celkové skóre za úlohu (max 100 bodů) se přidělí normalizovaně:

$$S = \frac{\text{RMSE}_{\text{worst}} - \text{RMSE}_{\text{váš}}}{\text{RMSE}_{\text{worst}} - \text{RMSE}_{\text{best}}} \times 100$$

kde `worst` a `best` odpovídají nejhorší a nejlepší hodnotě RMSE mezi všemi platnými odevzdáními v daném kole. Pokud je odevzdáno pouze jedno platné řešení, získá 100 bodů.

---

## 8. Odevzdání

| # | Soubor | Popis |
|---|---|---|
| 1 | `submission.csv` | Predikce pro všechny body z `test_points.csv` |
| 2 | `solution.ipynb` | Kompletní řešení — trénink, vizualizace, export CSV |

Odevzdání probíhá na platformě **platform.aiolympiada.cz**.

### Formát `submission.csv`

```csv
id,u
0,0.7523
1,0.4891
2,0.6234
...
```

> **⚠️ Důležité:**
>
> - Soubor musí obsahovat predikci pro **všechny** body z `test_points.csv`.
> - Chybějící řádky budou hodnoceny jako maximální chyba.
> - Sloupec se jmenuje `u`, nikoli `temperature` (bezrozměrné jednotky).

### Notebook `solution.ipynb`

Notebook musí být **spustitelný** a obsahovat:

- kompletní trénink modelu,
- generování `submission.csv`,
- alespoň jeden graf (např. predikovaný teplotní profil nebo průběh loss).

Notebook slouží jako důkaz originality řešení. V případě podezření na plagiát nebo generované řešení může organizátor požádat o ústní vysvětlení.

---

## 9. Pravidla a povolené nástroje

### 9.1 Prostředí

| | |
|---|---|
| **Prostředí** | Google Colab (Python 3, GPU runtime T4) |
| **Přístup k internetu** | Povolen pouze pro stažení dat ze soutěžní platformy a oficiální dokumentace knihoven |
| **Časový limit** | Notebook musí doběhnout do **30 minut** na Colab GPU T4 |

### 9.2 Povolené knihovny

Povoleny jsou knihovny **předinstalované v Google Colab**:

- PyTorch (včetně `torch.autograd`)
- NumPy, pandas, matplotlib, scikit-learn
- TensorFlow / Keras, JAX (volitelně)

**Zakázáno:**

- Specializované PINN frameworky (DeepXDE, NVIDIA Modulus, NeuralPDE apod.) — cílem úlohy je, aby studenti implementovali princip sami.
- Instalace dalších balíčků přes `pip install`.
- Předtrénované modely.

### 9.3 LLM

Použití LLM je povoleno výhradně přes **soutěžní rozhraní** (GPT-4o-mini, max 4 000 tokenů/dotaz). Použití vlastních API klíčů, jiných modelů nebo lokálních LLM je zakázáno.

### 9.4 Typ řešení

Řešení musí být postaveno na **neuronové síti**. Čistě numerická řešení PDE (finite differences, Crank–Nicolson, finite elements apod.) nejsou přípustná jako hlavní model. Fyzikální rovnici lze a je vhodné využít jako součást loss funkce.

---

## 10. Klíčové pojmy

| Pojem | Vysvětlení |
|---|---|
| **PINN** | Physics-Informed Neural Network — síť, jejíž loss zahrnuje i fyzikální rovnici |
| **Residual** | Míra porušení fyzikální rovnice modelem — čím menší, tím lépe |
| **Okrajová podmínka** | Známá hodnota teploty na hranici domény (kraje tyče) |
| **Počáteční podmínka** | Známý stav systému v čase $t = 0$ |
| **Kolokační body** | Náhodně vzorkované body v doméně, kde se vyhodnocuje fyzikální loss |
| **Automatická diferenciace** | Výpočet derivací modelu v PyTorch (`torch.autograd.grad`) |
| **RMSE** | Root Mean Squared Error — míra průměrné chyby predikce |

---

## 11. Rozšíření pro národní kolo

V národním kole může být úloha rozšířena o **inverzní problém**: hodnota $\alpha$ nebude známá a studenti ji budou odhadovat z dat jako naučitelný parametr sítě. Mohou být přidány i další komplikace (více materiálů, silnější šum, extrapolační testovací body).

---

*Organizátor: nvias, z.s. | www.aiolympiada.cz | info@nvias.org*
