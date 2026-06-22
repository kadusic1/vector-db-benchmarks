# NotebookLM Prompt — Prezentacija: Optimizacija vektorskih baza podataka
> **Autor prompta:** generisano na osnovu analize seminarskog rada  
> **Limit slajdova:** 10  
> **Format u NotebookLM-u:** Presenter Slides (kraći, jasniji za prezentaciju)  
> **Jezik prezentacije:** Bosanski/Hrvatski/Srpski  

---

## ⚙️ KORAK PO KORAK — Kako koristiti ovaj prompt u NotebookLM-u

1. Otvori [notebooklm.google.com](https://notebooklm.google.com)
2. Klikni **"Create new notebook"**
3. **Dodaj izvor (Source):** Uploadaj PDF seminarskog rada (`projekt.pdf`)
4. U desnom panelu klikni **ikonu olovke pored "Slide Decks"**
5. Odaberi format: **"Presenter Slides"** + dužina: **"Default"**
6. U polje **"Describe the slide deck you want to create"** — zalijepi PROMPT ispod
7. Klikni **Generate** i sačekaj nekoliko minuta
8. Nakon generisanja — koristi dugme **"Revise"** za fine-tuning pojedinih slajdova
9. Preuzmi kao **PPTX** za dodatno uređivanje

---

## 📋 GLAVNI PROMPT ZA NOTEBOOKLM

> ⬇️ Kopiraj SVE ispod ove linije i zalijepi u NotebookLM polje za opis decka

---

```
Ti si vrhunski dizajner prezentacija i profesor računarskih nauka. 
Koristeći ISKLJUČIVO priloženi seminarski rad, kreiraj akademsku prezentaciju
od TAČNO 10 slajdova za odbranu pred komisijom i profesorom na Politehničkom
fakultetu Univerziteta u Zenici.

TEMA: Optimizacija vektorskih baza podataka — empirijska evaluacija indeksnih
struktura i kvantizacijskih tehnika.

PUBLIKA: Profesor (v. prof. dr. Denis Čeke) i studenti smjera Softversko
inženjerstvo, II ciklus. Pretpostavlja se solidno predznanje iz baza podataka
i mašinskog učenja.

KLJUČNI ZAHTJEVI ZA SVAKI SLAJD:
- Naslov slajda mora biti TVRDNJA ili PITANJE (npr. "HNSW postiže 2× nižu
  latenciju od IVF-a" — NE samo "Rezultati")
- Maksimalno 4 bullet poenta po slajdu, svaki bullet max 12 riječi
- Svaki slajd nosi JEDNU centralnu poruku
- Gdje god je moguće, koristi VIZUAL (grafikon, diagram, tabela) umjesto teksta
- Boja akcenat: tamno plava (#1a3a5c) za naslove, bijela pozadina, 
  svijetlo siva za podsadržaj
- Stil: čist, minimalistički, akademski — BEZ clip-arta, BEZ dekorativnih ikona

STRUKTURA PREZENTACIJE (10 slajdova — FIKSNO, ne mijenjaj redoslijed):

--- SLAJD 1: NASLOVNA STRANICA ---
Sadržaj:
- Glavni naslov: "Optimizacija vektorskih baza podataka"
- Podnaslov: "Empirijska evaluacija indeksnih struktura i kvantizacijskih tehnika"
- Student: Adi Kadušić, II-134
- Predmet: Optimizacija baza podataka | Profesor: v. prof. dr. Denis Čeke
- Studij: Softversko inženjerstvo, II ciklus studija
- Institucija: Politehnički fakultet, Univerzitet u Zenici
- Datum: Juni 2026.
Vizual: Apstraktna ilustracija višedimenzionalnog vektorskog prostora ili
čvorova grafa (plava paleta)

--- SLAJD 2: MOTIVACIJA — "Zašto milijardu vektora gazi standardne baze?" ---
Poruka: Eksplozivni rast AI sistema stvorio je inžinjerski problem bez
presedana u pohrani i pretrazi vektora.
Sadržaj (max 4 bulleta):
• 768-dimenzionalni float32 vektor = 3 KB; 1 milijarda vektora = >3 TB
• Iscrpna pretraga skalira s O(n) — nepraktična za web-razmjerne kolekcije
• "Prokletstvo dimenzionalnosti" čini klasične indekse neupotrebljivim
• Rješenje: aproksimativne metode (ANN) koje biraju kompromis brzina↔tačnost
Vizual: Ikonografska usporedba "brute force" vs. "ANN" pretrage

--- SLAJD 3: ČETIRI KONFIGURACIJE POD LUPOM ---
Poruka: Testiramo dvije dimenzije optimizacije — tip indeksa i kompresija
vektora — kroz četiri konfiguracije.
Sadržaj: Prikaži kao 2×2 tabelu ili četiri kartice:

| Konfiguracija | Tip indeksa    | Kompresija   | Alat   |
|---------------|----------------|--------------|--------|
| Flat          | Iscrpna pretraga | Bez kompresije | FAISS  |
| IVF           | Klaster-baziran  | Bez kompresije | FAISS  |
| IVF-PQ        | Klaster-baziran  | Produktna kvantizacija | FAISS |
| HNSW          | Graf-baziran     | Bez kompresije | Qdrant |

Ispod tabele: "3 istraživačke hipoteze • 500.000 pasusa • 1.000 upita
po konfiguraciji"

--- SLAJD 4: KAKO RADE HNSW I IVF? ---
Poruka: HNSW i IVF koriste fundamentalno različite strategije za
ograničavanje prostora pretrage.
Sadržaj (split dizajn — lijevo HNSW, desno IVF):
LIJEVO — HNSW:
• Višeslojna grafovska hijerarhija (sloj 0 = svi vektori)
• Pohlepna navigacija od vrha prema dnu: O(log n)
• Parametri: M=16 (veze/čvor), ef_construct=200, ef_search=128
DESNO — IVF:
• k-means dĳeli prostor na K=256 klastera (Voronoi ćelije)
• Pretraga samo u nprobe=32 najbližih klastera
• Smanjuje usporedbe sa O(n) na O(n·nprobe/K)
Vizual: Reprodukcija/inspiracija dijagrama HNSW slojevite strukture iz rada
(Slika 2.2)

--- SLAJD 5: EKSPERIMENTALNI POSTAV ---
Poruka: Svi eksperimenti provedeni su pod identičnim uvjetima na
standardizovanom benchmarku.
Sadržaj (kompaktna info-grafika):
📊 DATASET: MS MARCO v2.1 — 500.000 pasusa | 1.000 upita
🤖 MODEL: all-mpnet-base-v2 → 768-dim vektori, L2-normalizovani
💻 HARDVER: Intel i7-14650HX | 32 GB RAM | RTX 4070 Laptop | Ubuntu 24.04
📏 METRIKE: Latencija (ms) | Recall@10 | Memorijski otisak (MB)
🔬 STATISTIKA: 5.000 mjerenja/konfiguraciji | Wilcoxon signed-rank test

--- SLAJD 6: REZULTAT — "IVF-PQ je 47× brži od Flat-a, ali po cijenu tačnosti" ---
Poruka: Postoji jasan spektar brzine, ali brzina i tačnost su u direktnoj
suprotnosti.
Sadržaj: Prikaži tabelu rezultata (ČITKA, uvećana, bez viška):

| Konfiguracija | Mean (ms) | Recall@10 | P95 (ms)  |
|---------------|-----------|-----------|-----------|
| Flat          | 52,36     | 1,0000    | 54,6      |
| IVF           | 8,58      | 0,9564    | 9,5       |
| **HNSW**      | **7,25**  | **0,9972**| **44,1 ⚠️** |
| IVF-PQ        | 1,12      | 0,6679    | 1,2       |

Podvuci ⚠️ symbol uz P95 HNSW i dodaj bilješku: "HNSW: medijana sjajna,
rep distribucije problematičan"

--- SLAJD 7: REZULTAT — "IVF-PQ komprimira 40× — ali memorija nije jedini
trošak" ---
Poruka: Produktna kvantizacija donosi dramatičnu memorijsku uštedu, ali
uz 33 procentna poena gubitka tačnosti u ovoj konfiguraciji.
Sadržaj:
Vizual 1 (bar chart — memorija) **KORISTI SLIKU IZ source: memory_comparison.pdf**:
  Flat=1.464,8 MB | IVF=1.469,4 MB | IVF-PQ=35,8 MB ✅ | HNSW=2.072,0 MB ❌
Vizual 2 (ikona/tekstualni komentar):
  "IVF-PQ: kompresija 40,88× ✅ | pad tačnosti 33,21 pp ❌"
  "Agresivni parametri: M=64 pod-vektora, 8 bita — svaki potprostor
  akumulira grešku"
  "Rješenje: smanjiti M ili povećati n_bits"

--- SLAJD 8: KLJUČNI NALAZ — "HNSW dominira na Recall-Latency krivulji" ---
Poruka: Sweep parametara otkriva da HNSW nudi neoborivo superiorni
kompromis kada memorija nije ograničena.
Sadržaj:
Vizual  **KORISTI SLIKU IZ SOURCE: tradeoff_curve.pdf** (reprodukcija/inspiracija Slike 4.3 iz rada — Recall-Latency Tradeoff):
  IVF: nprobe 8→128, recall 0,88→0,99, latencija 2,25→33,79 ms (linearna)
  IVF-PQ: nprobe 8→128, recall zaglavljeno 0,64–0,67 (PQ plafon!)
  HNSW: ef_search 50→400, recall 0,986→0,999, latencija 6→9,5 ms (strmina!)
Ključna opservacija (bold): "IVF-PQ recall je ograničen kvantizacijskom
greškom — više nprobe ne pomaže!"

--- SLAJD 9: VERIFIKACIJA HIPOTEZA ---
Poruka: Dvije od tri hipoteze potvrđene — djelimično odbijanje H2 otkriva
važno praktično ograničenje.
Sadržaj (vizualni trafik-light):
🟢 H1 POTVRĐENA
   HNSW medijana 4,45 ms < IVF medijana 8,56 ms
   Wilcoxon: W=52.432, p≈0 | Recall@10=0,9972 > 90%

🟡 H2 DJELIMIČNO POTVRĐENA
   Kompresija 40,88× ✅ (zahtjev: ≥4×)
   Pad tačnosti 33,21 pp ❌ (zahtjev: <5 pp)
   Uzrok: agresivni M=64, previsoka kvantizacijska greška

🟢 H3 POTVRĐENA
   IVF-PQ = 35,8 MB ✅ (ispod 2 GB budžeta)
   HNSW = 2.072 MB ❌ (premašuje za samo 24 MB!)

--- SLAJD 10: ZAKLJUČAK — "Nema univerzalnog pobjednika — kontekst odlučuje" ---
Poruka: Svaka konfiguracija je optimalna za drugačiji set ograničenja —
pravi izbor ovisi o prioritetu aplikacije.
Sadržaj (decision framework — 3 scenarija):
┌─────────────────────────────────────────────────────┐
│ Prioritet: Tačnost       → HNSW                     │
│ (bez memorijskog budžeta)   Recall 0,997 | 4,45 ms  │
├─────────────────────────────────────────────────────┤
│ Prioritet: Konzistentnost → IVF                     │
│ (predvidivi P95, low jitter) Recall 0,956 | P95=9ms │
├─────────────────────────────────────────────────────┤
│ Prioritet: Memorija       → IVF-PQ                  │
│ (<2 GB budžet)              35,8 MB | 1,12 ms       │
└─────────────────────────────────────────────────────┘
Donje: "Buduće smjernice: testirati HNSW u FAISS (bez mrežnog overheada) |
fine-tune PQ parametara | evaluacija ScaNN i DiskANN"

OPĆE NAPOMENE ZA DIZAJN:
- Koristi DARK BLUE (#1a3a5c) za naslove slajdova
- Bijela (#ffffff) pozadina
- Accent boja za ključne nalaze: zlatno-narandžasta (#e87722)
- Font: moderan sans-serif (Calibri, Inter ili slično), min 20pt za body
- Svaki slajd ima FOOTER sa: "Adi Kadušić | Politehnički fakultet UNZE | 2026."
- Tablice: zebra-striping (naizmjenično bijela/svjetlo siva)
- Boldiraj sve ključne numeričke nalaze
- NE koristiti animacije — statički slajdovi za čitku PPTX prezentaciju
```

---

## 🔧 PROMPTS ZA REVIZIJU POJEDINIH SLAJDOVA

Kada NotebookLM generira deck, klikni **"Revise"** na specifičnom slajdu i koristi ove prompts:

### Za Slajd 6 (tabela rezultata):
```
Make the results table cleaner and more readable. Bold the HNSW row.
Add a warning symbol (⚠️) next to the P95 value for HNSW to highlight
the tail latency issue. Keep it academic and minimal.
```

### Za Slajd 8 (Recall-Latency krivulja):
```
This slide needs a visual tradeoff chart. Show three lines/curves:
IVF (rising linearly), IVF-PQ (flat, stuck around 0.65 recall),
and HNSW (steep rise to 0.99 with minimal latency cost).
The key insight to highlight: IVF-PQ is fundamentally limited by
quantization error, not search scope.
```

### Za Slajd 9 (hipoteze):
```
Use a traffic light visual (green/yellow/green). Each hypothesis
gets one row with: hypothesis label, verdict icon, and key numbers
that prove or disprove it. Keep it scannable — this slide should
be understood in 10 seconds.
```

### Za Slajd 10 (zaključak):
```
Replace bullet points with a 3-row decision table:
Column 1: "Ako je prioritet..." | Column 2: "Odaberi..." 
Column 3: "Ključni broj"
Make it feel like a practical engineering cheat sheet.
```

---

## 📝 NAPOMENE ZA PREZENTACIJU (govorni tekst po slajdovima)

Ove napomene koristi kao **Speaker Notes** — dodaj ih ručno u PPTX nakon preuzimanja.

|
 Slajd 
|
 Ključna rečenica za uvod 
|

|
-------
|
--------------------------
|

|
 1 
|
 "Ovaj rad nastoji odgovoriti na jedno praktično inženjersko pitanje: kako pohraniti i pretraživati milion vektora brzo, tačno i jeftino — i šta se dogodi kad pokušate sve tri stvari odjednom." 
|

|
 2 
|
 "Jedan 768-dimenzionalni vektor zauzima 3 KB. To zvuči bezazleno — dok ne pomnoži sa milijardom zapisa i ne dobijete 3 terabajta. Niti jedan standardni server to ne podnosi u RAM-u." 
|

|
 3 
|
 "Testirali smo četiri strategije koje pokrivaju cijeli spektar od 'daj mi tačnost bez ograničenja' do 'daj mi brzinu bez ograničenja na memoriju'." 
|

|
 4 
|
 "HNSW i IVF se razlikuju kao GPS navigacija i karta po okruzima — oba vas dovode do cilja, ali drugačijim putem i s drugačijim kompromisima." 
|

|
 5 
|
 "MS MARCO je standardni benchmark za information retrieval. Koristimo ga jer ima stvarne web upite — ne sintetičke — što rezultatima daje praktičnu relevantnost." 
|

|
 6 
|
 "Pogledajte P95 za HNSW: 44 ms. To nije greška. HNSW ima izvrsnu medijanu, ali rep distribucije je problem koji bi u produkcijskom sistemu mogao biti kritičan." 
|

|
 7 
|
 "IVF-PQ postiže kompresiju 40 puta — to je impresivno. Ali cijena je 33 procentna poena tačnosti. Zašto toliko? Jer dijelimo vektor na 64 potprostora, a svaki unosi svoju grešku." 
|

|
 8 
|
 "Ovo je centralni nalaz rada. Pogledajte IVF-PQ krivulju — bez obzira koliko povećamo nprobe, recall ostaje zaglavljen oko 0.65. Kvantizacijska greška je plafon koji niti jedno podešavanje parametara pretrage ne može premostiti." 
|

|
 9 
|
 "H2 je djelimično odbijena — i to je možda najvažniji nalaz. Memorijsku kompresiju smo postigli s velikom rezervom, ali tačnost je žrtvovana previše agresivnim parametrima. Ovo je smjernica za buduće istraživanje." 
|

|
 10 
|
 "Zaključak nije 'HNSW je najbolji'. Zaključak je: pravi izbor ovisi o tome šta vas boli više — RAM, kašnjenje, ili tačnost. Ovaj rad daje empirijsku osnovu za tu odluku." 
|


---

## ⚠️ MOGUĆE GREŠKE I RJEŠENJA

|
 Problem 
|
 Uzrok 
|
 Rješenje 
|

|
---------
|
-------
|
----------
|

|
 Generisano više od 10 slajdova 
|
 NotebookLM ignoriše limit 
|
 U "Revise" napiši: "Merge slide X and Y into one. Keep only the most important point." 
|

|
 Slajdovi previše tekstualni 
|
 Default ponašanje 
|
 Dodaj u prompt: "Reduce text by 50%. Convert all bullet lists to visual elements where possible." 
|

|
 Naslovi nisu tvrdnje 
|
 Generički slajd naslovi 
|
 U "Revise" svakog slajda: "Rewrite title as an assertion or finding, not a noun label." 
|

|
 Tabele nečitljive 
|
 Premalen font u tablicama 
|
 U PPTX ručno povećaj font tablice na min 18pt 
|

|
 Krivulja sa Slajda 8 nije generisana 
|
 NotebookLM ne može reproducirati grafikon iz rada 
|
 Eksportuj u PPTX i ručno umetni screenshot grafikona iz PDF-a 
|


---

## 🎯 STRATEGIJA ZA ODBRANU (bonus)

**Pitanja koja profesor vjerovatno postavlja i odgovori:**

**P: "Zašto HNSW niste testirali direktno u FAISS-u bez mrežnog overheada?"**  
O: "Svjesni smo ovog ograničenja. Qdrant je odabran jer je produkcijski servis za koji je HNSW optimiziran, što rezultatima daje praktičnu relevantnost. Mrežni overhead od 0,67 ms izmjeren je i oduzet na nivou medijane pri apsolutnim usporedbama, no visoka varijabilnost u repovima može biti artefakt testnog okruženja — ne isključivo karakteristika algoritma. Ovo je eksplicitno navedeno kao ograničenje i smjernica za buduće istraživanje."

**P: "Kako biste poboljšali H2 da bude u potpunosti potvrđena?"**  
O: "Smanjili bismo broj pod-vektora M (npr. sa 64 na 16 ili 32) ili povećali n_bits na 12 ili 16. Time bismo smanjili akumuliranu kvantizacijsku grešku, vjerovatno zadovolivši uvjet od <5 procentnih poena pada tačnosti — ali uz manji kompresijski omjer."

**P: "Šta je Wilcoxon test i zašto ste ga koristili umjesto t-testa?"**  
O: "Wilcoxon signed-rank test je neparametrijski test koji ne pretpostavlja normalnu distribuciju podataka. Distribucije latencija su tipično asimetrične s dugim desnim repom — što je i ovdje slučaj, posebno za HNSW. t-test bi bio neprikladniji jer pretpostavlja normalnost, pa bi bio narušen ovim repovima distribucije."

---

*Prompt kreiran na osnovu analize seminarskog rada "Optimizacija vektorskih baza podataka" (Adi Kadušić, PFUNZE, 2026) uz primjenu istraživačkih principa tehničkog prezentiranja (MIT EECS CommLab, Harvard Catalyst) i best practice-a za NotebookLM Slide Decks.*

Na osnovu 4 source-a napravi prezentaciju od 10 slajdova! Projekt.pdf je seminarski rad a vector db copied text su SYSTEM PROMPT INSTRUKCIJE KOJE MORAS POSTOVATI! JEZIK NEKA BUDE BOSANSKI (IAKO JE OVDJE NAVEDEN HRVATSKI) I FORMAT PRESENTER SLIDES! OBAVEZNO U RADU KORISTI SLIKE memory_comparison.pdf i tradeoff_curve.pdf PREMA INSTRUKCIJAMA IZ SYSTEM PROMPTA!