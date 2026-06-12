# PROJEKAT: Optimizacija vektorskih baza podataka

Benchmarking vektorskih indeksa za pretragu slicnosti u prostoru
visoke dimenzije. Poredi cetiri tipa indeksa na MS MARCO datasetu
(500k pasusa, 768-dim embeddinga): Flat (brute-force), IVF,
IVF-PQ i HNSW. Mjeri latenciju, recall@10 i memorijski otisak,
te statisticki testira tri hipoteze Wilcoxon signed-rank testom.

## Struktura projekta

```
database_optimization/
+-- main.py                     Ulazna tacka: pokrece pipeline
+-- Dockerfile                  Python 3.13 image sa uv
+-- docker-compose.yml          Qdrant + app servis
+-- pyproject.toml              Zavisnosti (bez torch/ST)
+-- .env.example                Template za .env
+-- .dockerignore               Sprecava data/ u build contextu
+-- src/
|   +-- pipeline.py             Orkestracija 8 faza
|   +-- data.py                 MS MARCO loader
|   +-- embeddings.py           SentenceTransformer engine
|   +-- benchmark.py            Benchmark + verifikacija hipoteza
|   +-- metrics.py              Recall@10, medijane, memorija
|   +-- statistics.py           Wilcoxon signed-rank test
|   +-- plots.py                Grafici (tradeoff, boxplot, bar)
|   +-- logger.py               Logging setup
|   +-- core/
|   |   +-- index.py            ABC za vektorski indeks
|   +-- indices/
|       +-- flat.py             FAISS IndexFlatL2
|       +-- ivf.py              FAISS IndexIVFFlat
|       +-- ivfpq.py            FAISS IndexIVFPQ
|       +-- hnsw.py             Qdrant HNSW
+-- scripts/
|   +-- create_embeddings.py    Generisanje embeddinga (opciono)
+-- output/
    +-- artifacts/              Serijalizovani FAISS indeksi
    +-- figures/                Generisani grafici
```

## Preduvjeti

* Docker + Docker Compose
* ~2.5 GB slobodnog prostora
* Internet za preuzimanje data fajlova

## Preuzimanje podataka

Aplikacija zahtijeva 11 Parquet fajlova u `data/` folderu.
Preuzeti ih sa HuggingFace Datasets-a:

1. Otvoriti https://huggingface.co/datasets/kadusicadi/ms-marco-embeddings
2. Otići na `files_and_versions`
3. Skinuti sve fajlove u lokalni `data/` folder (kreirati ga ako
   ne postoji)
4. **Embeddingsi** su podijeljeni na više fajlova radi lakšeg preuzimanja. Ukupna
   veličina embeddingsa za preuzimanje iznosi `1.65 GB`. Konkretno ovdje ima 11
   `parquet` fajlova koji se trebaju preuzeti:

   ```
   data/
   +-- passages_0.parquet
   +-- passages_1.parquet
   +-- ...
   +-- passages_9.parquet
   +-- queries.parquet
   ```

## Alternativa preuzimanju podataka NIJE PREPORUČENO - generisanje embeddinga

Umjesto preuzimanja gotovih embeddinga sa HuggingFace-a, moguce je
generisati ih lokalno. **Ovo se ne preporucuje** -- detalji su
navedeni u nastavku.

### Kako pokrenuti

Embedding engine zahtijeva `sentence-transformers` i `torch`, koji
nisu ukljuceni u osnovne zavisnosti. Potrebno ih je instalirati
rucno (ne kroz Docker):

```bash
pip install -e ".[generate]"
```

Zatim pokrenuti skriptu:

```bash
python scripts/create_embeddings.py
```

Ovo **ne radi kroz Docker** -- enkodiranje 500k pasusa zahtijeva
pristup GPU-u i vise RAM-a nego sto je docker container-u
dodijeljeno. Skripta se pokrece direktno na host masini (ili
u virtuelnom okviru sa GPU propustanjem).

### Zasto se ne preporucuje

1. **Vrijeme** -- enkodiranje 500k pasusa modelom
   `all-mpnet-base-v2` traje ~1 sat na GPU-u, a visestruko duze
   na CPU-u.
2. **Hardver** -- preporucuje se GPU sa >=8 GB VRAM; na CPU-u je
   potrebno 16+ GB RAM uz znacajno duze vrijeme.

## Pokretanje

```bash
cp .env.example .env
docker compose up --build
```

Prvi build traje ~60 sekundi. Nakon toga app ceka da Qdrant
postane zdrav, pa pokrece pipeline od 8 faza.

## Sta se desava (8 faza)

1. **Ucitavanje** -- 500k pasusa + 1000 upita sa diska
2. **Izgradnja indeksa** -- Flat (brute-force), IVF, IVF-PQ,
   HNSW (u Qdrant-u)
3. **Mrezni overhead** -- mjerenje Qdrant RTT-a
4. **Benchmark** -- svaki upit 5 puta kroz svaki indeks
5. **Metrike** -- recall@10, medijane latencije
6. **Statistika** -- deskriptivna + Wilcoxon test za H1, H2, H3
7. **Sweep parametara** -- nprobe (IVF/IVF-PQ), ef_search (HNSW)
8. **Grafovi** -- tradeoff krive, boxplot, memorijski bar chart

Rezultati se upisuju u `output/figures/` i `output/artifacts/`.
