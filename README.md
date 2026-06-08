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
2. Skinuti sve fajlove u lokalni `data/` folder (kreirati ga ako
   ne postoji):

   ```
   data/
   +-- passages_0.parquet
   +-- passages_1.parquet
   +-- ...
   +-- passages_9.parquet
   +-- queries.parquet
   ```

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
