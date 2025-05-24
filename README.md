# UJEP Hackathon 2025

This repository was created by a team of 4 people in 24 hours during anual UJEP Hackathon.

## Contest task
Design and implement arterial pressure generator with anomalies.

## Setup
```bash
git clone https://github.com/ondrejsvorc/Hackathon-2025.git
cd Hackathon-2025
py -m venv .venv
pip install -r requirements.txt
```

Add `data` folder to the root project directory. Download data [here](http://data.ki.ujep.cz/files/Generov%C3%A1n%C3%AD%20syntetick%C3%BDch%20sign%C3%A1l%C5%AF%20s%20anom%C3%A1liemi/).

## Generování syntetických signálů s anomáliemi
**Tým:** Vlnky
**Hackathon:** CreaThon 2025
**Zadání:** C – Generování syntetických signálů arteriálního tlaku s anomáliemi

## Úvod

Cílem naší práce bylo navrhnout a implementovat generátor realistických fyziologických signálů arteriálního tlaku, které budou:
- zcela syntetické,
- věrohodně obsahovat anomálie,
- parametrizovatelné,
- využitelné pro trénink a testování modelů detekce anomálií.

## Analýza dat

Dataset (`.hdf5`) obsahuje 90 hodin reálných, expertně anotovaných záznamů arteriálního tlaku. Pro jeho prohlížení jsme využili dostupné nástroje:

- [HDF5 Visualizer](https://pavelfalta.github.io/hdf5visualizer/)
- [Repozitář s nástroji](https://github.com/PavelFalta/creathon25)

---

## Použité metody

### 1. VAE na raw signálech
- První pokus: **Variational Autoencoder (VAE)** na původních časových signálech.
- Cíl: získat kompaktní latentní prostor pro generování nových signálů.

### 2. Transformers
- Vyzkoušeli jsme architekturu **Transformers**, která zvládá dlouhodobé závislosti.
- Výsledky byly nadějné, zejména při kombinaci s latentními reprezentacemi.

### 3. Kombinace VAE + Transformers
- VAE pro extrakci latentních reprezentací, Transformers pro generování v čase.
- Přineslo vyšší kvalitu syntetických dat než jednotlivé přístupy zvlášť.

### 4. Přechod do frekvenční domény
- Pomocí **Fourierovy transformace** jsme převedli data do frekvenční oblasti.
- VAE trénovaný na spektrech -> rekonstrukce zpět do časové oblasti.
- Snížila se náhodná oscilace a zvýšila se stabilita výstupů.

### 5. GAN framework
- Experiment s **Generative Adversarial Networks (GAN)**.
- Generování realistických 5s úseků signálů.
- Omezení: malá délka výstupu, ale výborná věrohodnost.

## Webová prezentace

Vytvořili jsme interaktivní web pro vizualizaci vygenerovaných signálů:

- SVG model lidského těla (např. kliknutí na „hlavu“)
- Animovaný graf zobrazující signál v čase
- Přístupné i nespecialistům

## Parametrizace generátoru

Naše řešení umožňuje generování signálů s různými parametry:

- **Pulzní frekvence** (údery za minutu)
- **Pulzní tlak**
- **Výskyt a typy anomálií**
- **Délka generovaného signálu**

## Metriky hodnocení
- :)

## Shrnutí

Během hackathonu jsme:

- Prozkoumali více přístupů k modelování signálu.
- Identifikovali silné a slabé stránky jednotlivých metod.
- Vytvořili funkční webovou aplikaci pro vizualizaci.
- Vyvinuli generátor, který lze parametrizovat a použít k tréninku modelů detekce anomálií.

> _Tento projekt byl vytvořen v rámci hackathonu [CreaThon 2025](https://creathon.cz) na zadání č. C – Generování syntetických signálů s anomáliemi._
