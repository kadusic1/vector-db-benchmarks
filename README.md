# Metodologija istraživanja: Optimizacija vektorskih baza podataka

Eksperimentalno-kvantitativno istraživanje — 4 konfiguracije indeksa, 1000 upita × 5 ponavljanja.

---

## Pregled

| Faza | Naziv | Opis |
|------|-------|------|
| 0 | Okruženje | Python, Qdrant, FAISS, Docker |
| 1 | Podaci | MS MARCO dataset + 1K upita |
| 2 | Embeddings | all-mpnet-base-v2 → 768D vektori |
| 3 | Indeksi | Flat · HNSW · IVF · IVF-PQ |
| 4 | Benchmarking | Latencija po konfiguraciji |
| 5 | Metrike | Recall@10, memorija, latencija |
| 6 | Statistika | Wilcoxon + trade-off krivulje |

### Hipoteze koje se testiraju

| Hipoteza | Poređenje | Prag |
|----------|-----------|------|
| H1 | HNSW vs IVF latencija | Recall@10 > 90% |
| H2 | IVF-PQ vs Flat memorija | ≥ 4× redukcija, ≤ 5pp recall pad |
| H3 | IVF-PQ vs HNSW trade-off | Memorija < 2 GB |

### Tehnološki stack

`Qdrant (HNSW)` · `FAISS (IVF, IVF-PQ, Flat)` · `sentence-transformers` · `MS MARCO v1.1` · `scipy (Wilcoxon)` · `Python 3.10+` · `Docker`

---

## Zašto i Qdrant i FAISS?

Ovo je ključno arhitekturalno pitanje. Razlog je metodološki — svaki alat je optimiziran za drugu klasu indeksa.

> **Kratki odgovor:** Qdrant je optimiziran za HNSW u produkcijskom okruženju (Rust, HTTP/gRPC), a FAISS je referentna biblioteka za IVF i Product Quantization. Kombinacijom dobijamo svaki indeks testiran u svom "optimalnom" okruženju — što daje validnije rezultate.

### Podjela odgovornosti

| Konfiguracija indeksa | Alat | Razlog |
|-----------------------|------|--------|
| Flat (L2) | FAISS — `IndexFlatL2` | Brute-force baseline; direktna biblioteka, nema overhead-a |
| IVF | FAISS — `IndexIVFFlat` | Referentna IVF implementacija; precizna kontrola `nlist`/`nprobe` |
| IVF-PQ | FAISS — `IndexIVFPQ` | Product Quantization je core FAISS feature; Qdrant ga ne eksponira direktno |
| HNSW | Qdrant kolekcija | Qdrant je napisan u Rustu specijalno za HNSW; simulira produkcijski deploy |

### FAISS — za šta je dobar?

- Radi in-process (Python) → minimalan overhead
- Fine-grained kontrola: `nlist`, `nprobe`, `M`, `nbits`
- Akademski standard za ANN benchmarke
- Podržava GPU akceleraciju
- IVF trening (k-means clustering)
- PQ kompresija vektora za H2 hipotezu

### Qdrant — za šta je dobar?

- Native HNSW optimiziran u Rustu
- Persistentno skladištenje (realni deploy)
- HTTP/gRPC API (simulira mrežnu latenciju)
- Podržava Scalar Quantization nativno
- Filtriranje po metapodacima
- HNSW parametri: `m`, `ef_construct`, `ef_search`

### Zašto ne koristiti samo FAISS?

FAISS podržava HNSW (`IndexHNSWFlat`), ali ta implementacija nije optimizirana za produkcijsko okruženje. Qdrant-ov HNSW (napisan u Rustu, s async I/O, SIMD optimizacijama i persistencijom) relevantniji je za realnu upotrebu — što je i predmet istraživanja: koji indeks je bolji **u praksi**.

### Zašto ne koristiti samo Qdrant?

Qdrant ne eksponira direktno IVF-PQ konfiguraciju na isti način kao FAISS. Qdrant ima vlastiti pristup kvantizaciji (Scalar i Product Quantization kao overlay nad HNSW-om), ali je IVF-PQ kao samostalni indeks karakteristična FAISS paradigma. Za testiranje H2 i H3 hipoteza potreban je puni kontrolirani pristup PQ parametrima — što FAISS pruža.

### Zajednička validacija (H3 hipoteza)

H3 direktno poredi IVF-PQ (FAISS) sa HNSW (Qdrant) uz memorijsko ograničenje. Ovo je cross-tool poređenje koje je intentional — istraživanje mjeri koji *pristup* (kvantizacija+IVF vs grafovska struktura) daje bolji recall-latency trade-off, bez obzira na alat. Upravo zato što se koriste dva alata, rezultat ima praktičnu relevantnost: direktno odgovara na pitanje "Qdrant ili FAISS za memorijski ograničene sisteme?"

---

## Faza 0 — Priprema okruženja

Instaliraj sve potrebne alate i pokreni Qdrant servis.

### Korak 1: Instalacija Python paketa

Kreiraj virtualno okruženje i instaliraj sve zavisnosti.

```bash
# Kreiranje virtualnog okruženja
python -m venv venv && source venv/bin/activate

# Core paketi
pip install qdrant-client faiss-cpu sentence-transformers
pip install datasets numpy scipy pandas matplotlib tqdm
```

> **GPU napomena:** Za GPU zamijeni `faiss-cpu` s `faiss-gpu` ako imaš CUDA-kompatibilnu karticu — drastično ubrzava trening IVF indeksa.

### Korak 2: Pokretanje Qdrant servisa (Docker)

Qdrant se pokreće kao mikroservis. Eksponira REST (6333) i gRPC (6334) portove.

```bash
# Povuci i pokreni Qdrant kontejner
docker pull qdrant/qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant_bench \
  qdrant/qdrant

# Verifikacija (trebas dobiti { "title": "qdrant" })
curl http://localhost:6333/
```

### Korak 3: Hardverski zahtjevi

MS MARCO s 100k pasusa i 768D vektorima zahtijeva:

```
# Procjena memorije:
# - Flat  (100k × 768 × 4 bajta) = ~295 MB (float32)
# - HNSW  (graph overhead ≈ 2-4×) = ~600 MB – 1.2 GB
# - IVF   (vectors + codebook)    = ~310 MB
# - IVF-PQ (M=64, nbits=8)       = ~100k × 64 bajta  = ~6 MB
#   → 4× do 50× manje od Flat!
```

---

## Faza 1 — Priprema podataka

Učitaj MS MARCO dataset, izdvoji pasuse i 1000 standardiziranih upita.

### O MS MARCO datasetu

| Karakteristika | Vrijednost |
|----------------|------------|
| Puni dataset | ~8.8M pasusa, ~1M upita |
| Preporučeni subset | 100k–500k pasusa (ovisno o hardware-u) |
| Query set | 1000 upita iz validacijskog split-a |
| Format | Parovi pitanje–pasusi s relevance labelama |
| Verzija | v1.1 (Passage Retrieval) |

### Korak 1: Učitavanje dataseta

```python
from datasets import load_dataset

# Učitaj MS MARCO v1.1 (Passage retrieval zadatak)
dataset = load_dataset("microsoft/ms_marco", "v1.1")

# Struktura:
# dataset['train'] → ~808k primjera
# dataset['validation'] → ~101k primjera
# Svaki primjer: {'query', 'passages': {'passage_text', 'is_selected'}}
```

### Korak 2: Ekstrakcija pasusa i upita

```python
import random, numpy as np

# Izvuci sve pasuse iz train split-a (deduplikovano)
all_passages = []

for example in dataset['train']:
    for pid, p_text in enumerate(example['passages']['passage_text']):
        all_passages.append(p_text)

# Uzorkuj 100k pasusa (za razumno trajanje eksperimenta)
random.seed(42)
idx = random.sample(range(len(all_passages)), 100_000)
passages_subset = [all_passages[i] for i in idx]

# 1000 standardiziranih upita iz validation set-a
queries_1k = [ex['query'] for ex in dataset['validation']][:1000]

print(f"Pasusa: {len(passages_subset)}")   # 100,000
print(f"Upita: {len(queries_1k)}")          # 1,000
```

> **Standardizacija upita:** Isti 1000 upita koriste se za sve 4 konfiguracije. To je ključno — uklanja slučajni šum i osigurava da su poređenja statistički validna (Wilcoxon test radi na parovima).

---

## Faza 2 — Generisanje embeddings

Pretvori tekst u 768-dimenzionalne vektore koristeći all-mpnet-base-v2.

### O all-mpnet-base-v2 modelu

| Karakteristika | Vrijednost |
|----------------|------------|
| Arhitektura | MPNet (Masked and Permuted Pre-training) |
| Dimenzija vektora | 768 |
| Max sequence length | 514 tokena |
| Trenirano na | 1B+ rečenica (AllNLI, MS MARCO, ...) |
| Similarity metrika | Cosine similarity |
| Benchmark (SBERT) | Jedan od najpreciznijih sentence modela |

### Korak 1: Enkodiranje pasusa (batch)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Batch enkodiranje pasusa (batch_size ovisi o RAM/GPU)
print("Enkodiranje pasusa...")
passage_emb = model.encode(
    passages_subset,
    batch_size=128,           # prilagodi hardware-u
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # L2 normalizacija za cosine
)
# Shape: (100000, 768), dtype: float32

# Enkodiranje upita
query_emb = model.encode(
    queries_1k,
    batch_size=128,
    normalize_embeddings=True,
    convert_to_numpy=True
)
# Shape: (1000, 768)
```

### Korak 2: Zašto L2 normalizacija?

Normalizacijom vektora na jediničnu sferu, **Euclidean distance (L2)** i **Cosine similarity** postaju ekvivalentni. To nam omogućava da koristimo `IndexFlatL2` u FAISS-u dok efektivno mjerimo kosinusnu sličnost — konzistentno s Qdrant-ovim `Distance.COSINE`.

```python
# Verifikacija normalizacije
norms = np.linalg.norm(passage_emb, axis=1)
print(f"Norm range: [{norms.min():.4f}, {norms.max():.4f}]")
# Trebas dobiti: Norm range: [1.0000, 1.0000]

# Sačuvaj na disk (rekoristi za sve indekse)
np.save('passage_embeddings.npy', passage_emb)
np.save('query_embeddings.npy', query_emb)
```

> **Trajanje:** Enkodiranje 100k pasusa na CPU traje ~20-40 minuta, na GPU (T4 / RTX 3060) ~3-5 minuta. Preporučeno: sačuvaj na disk i reukoristi za sve eksperimente.

---

## Faza 3 — Izgradnja indeksa

Četiri konfiguracije: Flat (FAISS), IVF (FAISS), IVF-PQ (FAISS), HNSW (Qdrant).

### 1. Flat indeks — FAISS (baseline)

Brute-force pretraga — svaki upit poredi sa svim vektorima. 100% recall, maksimalna memorija, spora latencija. Služi kao gornja granica za recall i donja za brzinu.

```python
import faiss
dim = 768

# Flat L2 indeks (exact search)
index_flat = faiss.IndexFlatL2(dim)
index_flat.add(passage_emb.astype(np.float32))

print(f"Flat: {index_flat.ntotal} vektora")  # 100,000
# Memorija: ~295 MB (100k × 768 × 4 bajta)
```

### 2. IVF indeks — FAISS

Inverted File Index — dijeli vektore u `nlist` klastera k-means algoritmom. Pri pretrazi pregledava samo `nprobe` najbližih klastera. Trade-off: veći nprobe = bolji recall, veća latencija.

```python
nlist = 256  # broj klastera (tipično sqrt(N) do 4*sqrt(N))

# Kvantizator definira klaster centroide
quantizer = faiss.IndexFlatL2(dim)

# IVF indeks s flat storage-om unutar klastera
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)

# OBAVEZNO: IVF mora biti treniran (k-means nad podacima)
print("Trening IVF indeksa...")
index_ivf.train(passage_emb.astype(np.float32))

index_ivf.add(passage_emb.astype(np.float32))
index_ivf.nprobe = 32  # pregledaj 32/256 klastera pri pretrazi

# Sačuvaj za reupotrebu
faiss.write_index(index_ivf, "index_ivf.faiss")
```

> **nprobe sweep:** Za recall-latency krivulje, testiraj nprobe ∈ {8, 16, 32, 64, 128} — svaka vrijednost daje jednu tačku na krivulji.

### 3. IVF-PQ indeks — FAISS

Kombinira IVF klasterizaciju s Product Quantization kompresijom. Svaki 768D vektor dijeli se na `M` pod-vektora od 768/M dimenzija, svaki kvantiziran na `2^nbits` centroids. Rezultat: drastična ušteda memorije.

```python
# M mora dijeliti dim (768 % M == 0)
M = 64       # pod-vektori (svaki 768/64 = 12D)
nbits = 8    # 256 centroids po pod-vektoru

quantizer2 = faiss.IndexFlatL2(dim)
index_ivfpq = faiss.IndexIVFPQ(quantizer2, dim, nlist, M, nbits)

print("Trening IVF-PQ indeksa...")
index_ivfpq.train(passage_emb.astype(np.float32))
index_ivfpq.add(passage_emb.astype(np.float32))
index_ivfpq.nprobe = 32

# Memorija bez PQ: 100k × 768 × 4 = ~295 MB
# Memorija s PQ:  100k × 64  × 1 = ~6.4 MB  →  46× manje!
faiss.write_index(index_ivfpq, "index_ivfpq.faiss")
```

### 4. HNSW indeks — Qdrant

Hierarchical Navigable Small World graf. Qdrant gradi HNSW kao primary indeks. Parametri `m` (broj veza) i `ef_construct` (beam width pri gradnji) kontrolišu quality-speed trade-off.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, HnswConfigDiff, PointStruct
)

client = QdrantClient("localhost", port=6333)

# Kreiraj kolekciju s HNSW konfiguracijom
client.recreate_collection(
    collection_name="ms_marco_hnsw",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE  # L2-norm vektori → ekvivalentno s FAISS Flat L2
    ),
    hnsw_config=HnswConfigDiff(
        m=16,               # broj bidirektionalnih veza po čvoru
        ef_construct=200,   # beam width pri gradnji (viši = bolji ali sporiji build)
        full_scan_threshold=10_000  # fallback na brute-force za male kolekcije
    )
)

# Upload vektora (batch)
BATCH = 500
for i in range(0, len(passages_subset), BATCH):
    pts = [
        PointStruct(
            id=j,
            vector=passage_emb[j].tolist(),
            payload={"text": passages_subset[j]}
        )
        for j in range(i, min(i + BATCH, len(passages_subset)))
    ]
    client.upsert("ms_marco_hnsw", pts)

print("Upload završen.")
```

> **ef sweep za HNSW:** `ef_search` (query time parameter) varirati ∈ {50, 100, 200, 400} za recall-latency krivulju, koristeći `client.search(..., search_params=SearchParams(hnsw_ef=ef))`.

---

## Faza 4 — Benchmarking

1000 standardiziranih upita × 5 ponavljanja = 5000 mjerenja latencije po konfiguraciji.

### Korak 1: Benchmark funkcija za FAISS indekse

```python
import time
import numpy as np

def benchmark_faiss(index, query_embeddings, k=10, n_reps=5):
    """
    Vraća: listu 5000 latencija u milisekundama
    """
    # Warmup (izbjegni JIT/cache efekte u prvom pozivu)
    _ = index.search(query_embeddings[:10].astype(np.float32), k)

    latencies_ms = []
    results_all = []

    for rep in range(n_reps):
        for q_vec in query_embeddings:
            q = q_vec.reshape(1, -1).astype(np.float32)

            t0 = time.perf_counter()
            distances, indices = index.search(q, k)
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000)

            if rep == 0:  # Čuvaj rezultate samo iz prvog run-a
                results_all.append(indices[0].tolist())

    return latencies_ms, results_all  # (5000,), (1000,)
```

### Korak 2: Benchmark funkcija za Qdrant (HNSW)

```python
def benchmark_qdrant(client, query_embeddings, collection, k=10, n_reps=5, ef=128):
    from qdrant_client.models import SearchParams

    # Warmup
    client.search(collection, query_vector=query_embeddings[0].tolist(), limit=k)

    latencies_ms = []
    results_all = []

    for rep in range(n_reps):
        for q_vec in query_embeddings:
            t0 = time.perf_counter()
            hits = client.search(
                collection_name=collection,
                query_vector=q_vec.tolist(),
                limit=k,
                search_params=SearchParams(hnsw_ef=ef)
            )
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000)

            if rep == 0:
                results_all.append([h.id for h in hits])

    return latencies_ms, results_all
```

### Korak 3: Pokretanje svih konfiguracija

```python
# Ground truth: Flat indeks uvijek daje 100% recall
_, gt_results = benchmark_faiss(index_flat, query_emb, k=10, n_reps=1)

# Benchmark svih konfiguracija
configs = {}
configs['flat'],  flat_results  = benchmark_faiss(index_flat,  query_emb)
configs['ivf'],   ivf_results   = benchmark_faiss(index_ivf,   query_emb)
configs['ivfpq'], ivfpq_results = benchmark_faiss(index_ivfpq, query_emb)
configs['hnsw'],  hnsw_results  = benchmark_qdrant(client, query_emb, "ms_marco_hnsw")

print("Benchmark završen.")
for name, lats in configs.items():
    print(f"{name}: mean={np.mean(lats):.2f}ms, P95={np.percentile(lats,95):.2f}ms")
```

> **Važno — Qdrant uključuje HTTP overhead!** Svaki Qdrant query prolazi kroz TCP socket (localhost). Ovo je intentionally uključeno jer testiramo produkcijski scenario. Za čisto algoritamsko poređenje, latencije HNSW u Qdrant-u treba interpretirati u tom kontekstu.

---

## Faza 5 — Mjerenje metrika

Tri ključne metrike: latencija (ms), recall@10, memorijski footprint (MB).

### Recall@10

Mjeri koliko od top-10 rezultata tačnog (Flat) pretraživanja ANN metoda pronalazi. Ground truth su rezultati Flat indeksa.

```python
def recall_at_k(retrieved, ground_truth, k=10):
    """
    retrieved:    lista ID-ova koje je ANN vratio (top-k)
    ground_truth: lista ID-ova koje bi Flat vratio (top-k)
    Vraća: float [0.0, 1.0]
    """
    retrieved_set = set(retrieved[:k])
    gt_set        = set(ground_truth[:k])
    return len(retrieved_set & gt_set) / k

# Računaj recall za sve upite
def compute_mean_recall(ann_results, flat_results, k=10):
    recalls = [
        recall_at_k(ann_r, gt_r, k)
        for ann_r, gt_r in zip(ann_results, flat_results)
    ]
    return np.mean(recalls), recalls

r_ivf,   recalls_ivf   = compute_mean_recall(ivf_results,   gt_results)
r_ivfpq, recalls_ivfpq = compute_mean_recall(ivfpq_results, gt_results)
r_hnsw,  recalls_hnsw  = compute_mean_recall(hnsw_results,  gt_results)

print(f"IVF recall@10:    {r_ivf:.4f}")
print(f"IVF-PQ recall@10: {r_ivfpq:.4f}")
print(f"HNSW recall@10:   {r_hnsw:.4f}")
```

### Memorijski footprint

```python
import os, faiss

def faiss_index_size_mb(index_path):
    return os.path.getsize(index_path) / (1024**2)

def faiss_memory_mb_estimate(index):
    # Teoretska procjena za float32
    return (index.ntotal * index.d * 4) / (1024**2)

# FAISS indeksi: veličina serialized fajla
print(f"Flat:   {faiss_memory_mb_estimate(index_flat):.1f} MB")
print(f"IVF:    {faiss_index_size_mb('index_ivf.faiss'):.1f} MB")
print(f"IVF-PQ: {faiss_index_size_mb('index_ivfpq.faiss'):.1f} MB")

# Qdrant: via collections info API
info = client.get_collection("ms_marco_hnsw")
print(f"HNSW (Qdrant): {info.vectors_count} vektora")
# + provjeri disk: ls -lh qdrant_storage/
```

### Pregled metrika

| Metrika | Definicija | Jedinica | Hipoteza |
|---------|------------|----------|----------|
| Mean latencija | Prosječno vrijeme jednog upita | ms | H1, H3 |
| P95 latencija | 95. percentil latencija (worst-case) | ms | H1 |
| Recall@10 | Udio točnih rezultata vs Flat | [0, 1] | H1, H2, H3 |
| Memorija | Veličina indeksa na disku/RAM | MB / GB | H2, H3 |

---

## Faza 6 — Statistička analiza

Deskriptivna statistika + Wilcoxon signed-rank test + recall-latency trade-off krivulje.

### Korak 1: Deskriptivna statistika

```python
import pandas as pd
import numpy as np

def describe_latencies(name, latencies):
    arr = np.array(latencies)
    return {
        'konfiguracija': name,
        'mean_ms':   round(np.mean(arr), 3),
        'median_ms': round(np.median(arr), 3),
        'p95_ms':    round(np.percentile(arr, 95), 3),
        'p99_ms':    round(np.percentile(arr, 99), 3),
        'std_ms':    round(np.std(arr), 3),
        'min_ms':    round(np.min(arr), 3),
        'max_ms':    round(np.max(arr), 3),
    }

df_stats = pd.DataFrame([
    describe_latencies("Flat",   configs['flat']),
    describe_latencies("IVF",    configs['ivf']),
    describe_latencies("IVF-PQ", configs['ivfpq']),
    describe_latencies("HNSW",   configs['hnsw']),
])
print(df_stats.to_string(index=False))
```

### Korak 2: Wilcoxon signed-rank test (H1 hipoteza)

Neparametrijski test — ne pretpostavlja normalnu distribuciju latencija (koje su tipično right-skewed). Testira da li su dvije distribucije statistički različite na nivou p < 0.05.

```python
from scipy.stats import wilcoxon

# H1: HNSW ima statistički nižu latenciju od IVF?
# Koristimo 1000 mjerenja (1 per query, 1. repetition)
hnsw_lats_1k = configs['hnsw'][:1000]
ivf_lats_1k  = configs['ivf'][:1000]

stat, p_val = wilcoxon(hnsw_lats_1k, ivf_lats_1k, alternative='less')
# alternative='less' → H_A: HNSW latencija < IVF latencija

print(f"Wilcoxon stat={stat:.2f}, p={p_val:.6f}")
print(f"H1 prihvaćena: {p_val < 0.05 and r_hnsw > 0.90}")

# Poredbe za sve hipoteze:
# H1: hnsw_lats vs ivf_lats
# H2: ivfpq_lats vs flat_lats (latencija); + memorija ratio
# H3: ivfpq_lats vs hnsw_lats (uz provjeru memorija < 2GB)
```

### Korak 3: Recall-latency trade-off krivulje

Variraj parametar pretrage (`nprobe` za IVF/IVF-PQ, `ef_search` za HNSW) i plotiraj recall@10 vs mean latencija:

```python
import matplotlib.pyplot as plt

# IVF sweep: nprobe ∈ [8, 16, 32, 64, 128]
nprobe_vals = [8, 16, 32, 64, 128]
ivf_curve = []  # [(recall, latency), ...]

for np_val in nprobe_vals:
    index_ivf.nprobe = np_val
    lats, results = benchmark_faiss(index_ivf, query_emb, n_reps=3)
    rec, _ = compute_mean_recall(results, gt_results)
    ivf_curve.append((rec, np.mean(lats)))

# HNSW sweep: ef_search ∈ [50, 100, 200, 400]
ef_vals = [50, 100, 200, 400]
hnsw_curve = []

for ef in ef_vals:
    lats, results = benchmark_qdrant(client, query_emb, "ms_marco_hnsw", ef=ef)
    rec, _ = compute_mean_recall(results, gt_results)
    hnsw_curve.append((rec, np.mean(lats)))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(*zip(*ivf_curve),  'o-', label='IVF (nprobe sweep)')
ax.plot(*zip(*hnsw_curve), 's-', label='HNSW (ef sweep)')
ax.set_xlabel('Recall@10')
ax.set_ylabel('Mean latencija (ms)')
ax.set_title('Recall-Latency Trade-off')
ax.legend()
plt.savefig('tradeoff_curve.png', dpi=150, bbox_inches='tight')
```

> **Sažetak statističke obrade:** Za svaku od 3 hipoteze: (1) Wilcoxon test za latenciju, (2) direktna računica za memoriju i recall, (3) trade-off krivulja za vizualizaciju. Sve uz p < 0.05 prag i jasno navedenim effect size-om.
