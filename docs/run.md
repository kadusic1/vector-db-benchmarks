```bash
app-1   | [INFO] FAZA 1: Ucitavanje podataka sa diska
Generating train split: 500000 examples [00:02, 174140.56 examples/s]
app-1   | [INFO] Ucitano 500000 pasusa
Generating train split: 1000 examples [00:00, 23929.71 examples/s]
app-1   | [INFO] Ucitano 1000 embeddinga upita
app-1   | [INFO] FAZA 2: Izgradnja indeksa
app-1   | [INFO] Flat: izgradnja...
app-1   | [INFO] IVF: trening i izgradnja...
app-1   | [INFO] IVF-PQ: trening i izgradnja...
app-1   | [INFO] HNSW: izgradnja kolekcije u Qdrant-u...
app-1   | [INFO] Uploaded 50000/500000 vectors
app-1   | [INFO] Uploaded 100000/500000 vectors
app-1   | [INFO] Uploaded 150000/500000 vectors
app-1   | [INFO] Uploaded 200000/500000 vectors
app-1   | [INFO] Uploaded 250000/500000 vectors
app-1   | [INFO] Uploaded 300000/500000 vectors
app-1   | [INFO] Uploaded 350000/500000 vectors
app-1   | [INFO] Uploaded 400000/500000 vectors
app-1   | [INFO] Uploaded 450000/500000 vectors
app-1   | [INFO] Uploaded 500000/500000 vectors
app-1   | [INFO] FAZA 3: Mjerenje Qdrant mreznog overhead-a
app-1   | [INFO] Qdrant network overhead: 0.68 ms
app-1   | [INFO] FAZA 4: Benchmarking
app-1   | [INFO] Flat: benchmark...
app-1   | [INFO] IVF (nprobe=32): benchmark...
app-1   | [INFO] IVF-PQ (nprobe=32): benchmark...
app-1   | [INFO] HNSW (ef_search=128): benchmark...
app-1   | [INFO] FAZA 5: Metrike
app-1   | [INFO] Recall@10 - IVF: 0.9564, IVF-PQ: 0.6679, HNSW: 0.9961
app-1   | [INFO] FAZA 6: Statisticka analiza
app-1   | [INFO] Konfiguracija   Mean(ms)   Median(ms)   Std(ms)    P95(ms)    P99(ms)   
app-1   | [INFO] Flat            53.216     51.954       6.539      54.894     100.012   
app-1   | [INFO] IVF             8.671      8.651        0.598      9.612      10.005    
app-1   | [INFO] IVF-PQ          1.115      1.111        0.072      1.235      1.282     
app-1   | [INFO] HNSW            6.948      4.281        9.243      44.076     46.847    
app-1   | [INFO] Wilcoxon signed-rank testovi:
app-1   | [INFO] H1 (HNSW < IVF): statistic=51069.00, p=0.000000, significant=True
app-1   | [INFO] H1 (recall@10 > 90%): 0.9961 > 0.90 = True
app-1   | [INFO] FAZA 7: Sweep parametara
app-1   | [INFO] IVF nprobe=8: recall=0.8848, latency=2.2520 ms
app-1   | [INFO] IVF nprobe=16: recall=0.9307, latency=4.4122 ms
app-1   | [INFO] IVF nprobe=32: recall=0.9564, latency=8.6235 ms
app-1   | [INFO] IVF nprobe=64: recall=0.9790, latency=17.1134 ms
app-1   | [INFO] IVF nprobe=128: recall=0.9945, latency=34.3423 ms
app-1   | [INFO] IVF-PQ nprobe=8: recall=0.6448, latency=0.3134 ms
app-1   | [INFO] IVF-PQ nprobe=16: recall=0.6611, latency=0.5747 ms
app-1   | [INFO] IVF-PQ nprobe=32: recall=0.6679, latency=1.0849 ms
app-1   | [INFO] IVF-PQ nprobe=64: recall=0.6718, latency=2.1055 ms
app-1   | [INFO] IVF-PQ nprobe=128: recall=0.6741, latency=4.1898 ms
app-1   | [INFO] HNSW ef_search=50: recall=0.9872, latency=6.1228 ms
app-1   | [INFO] HNSW ef_search=100: recall=0.9943, latency=6.3991 ms
app-1   | [INFO] HNSW ef_search=200: recall=0.9979, latency=7.8214 ms
app-1   | [INFO] HNSW ef_search=400: recall=0.9991, latency=9.0303 ms
app-1   | [INFO] FAZA 8: Generisanje grafika
app-1   | [INFO] Generisan: output/figures/tradeoff_curve.png
app-1   | [INFO] Generisan: output/figures/boxplot_latency.png
app-1   | [INFO] Generisan: output/figures/memory_comparison.png
app-1   | [INFO] Verifikacija hipoteza
app-1   | [INFO] H2 (memorija >=4x): flat=1464.8MB / ivfpq=35.8MB = 40.88x >= 4x = True
app-1   | [INFO] H2 (recall drop < 5pp): drop=0.3321 < 0.05 = False
app-1   | [INFO] H3 (budzet < 2GB): IVFPQ=35.8MB < 2048MB = True
app-1   | [INFO] H3 (IVF-PQ bolji tradeoff od HNSW): True (HNSW=2139.1MB ne staje u budzet)
app-1   | [INFO] Sve hipoteze potvrdjene: False
app-1   | [INFO] Ukupno vrijeme simulacije: 0h 17m 56s
app-1   | [INFO] Sve faze zavrsene. Grafici su spremljeni u output/figures/.
```