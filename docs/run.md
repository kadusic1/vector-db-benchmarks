```bash
app-1   | [INFO] FAZA 1: Ucitavanje podataka sa diska
Generating train split: 500000 examples [00:03, 127931.06 examples/s]
app-1   | [INFO] Ucitano 500000 pasusa
Generating train split: 1000 examples [00:00, 25364.07 examples/s]
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
app-1   | [INFO] Qdrant network overhead: 0.67 ms
app-1   | [INFO] FAZA 4: Benchmarking
app-1   | [INFO] Flat: benchmark...
app-1   | [INFO] IVF (nprobe=32): benchmark...
app-1   | [INFO] IVF-PQ (nprobe=32): benchmark...
app-1   | [INFO] HNSW (ef_search=128): benchmark...
app-1   | [INFO] FAZA 5: Metrike
app-1   | [INFO] Recall@10 - IVF: 0.9564, IVF-PQ: 0.6679, HNSW: 0.9972
app-1   | [INFO] FAZA 6: Statisticka analiza
app-1   | [INFO] Konfiguracija   Mean(ms)   Median(ms)   Std(ms)    P95(ms)    P99(ms)   
app-1   | [INFO] Flat            52.357     51.708       2.840      54.638     69.415    
app-1   | [INFO] IVF             8.576      8.556        0.609      9.497      9.889     
app-1   | [INFO] IVF-PQ          1.117      1.113        0.158      1.235      1.282     
app-1   | [INFO] HNSW            7.252      4.451        9.382      44.118     47.566    
app-1   | [INFO] Wilcoxon signed-rank testovi:
app-1   | [INFO] H1 (HNSW < IVF): statistic=52432.00, p=0.000000, significant=True
app-1   | [INFO] H1 (recall@10 > 90%): 0.9972 > 0.90 = True
app-1   | [INFO] FAZA 7: Sweep parametara
app-1   | [INFO] IVF nprobe=8: recall=0.8848, latency=2.2461 ms
app-1   | [INFO] IVF nprobe=16: recall=0.9307, latency=4.3864 ms
app-1   | [INFO] IVF nprobe=32: recall=0.9564, latency=8.6004 ms
app-1   | [INFO] IVF nprobe=64: recall=0.9790, latency=16.9535 ms
app-1   | [INFO] IVF nprobe=128: recall=0.9945, latency=33.7881 ms
app-1   | [INFO] IVF-PQ nprobe=8: recall=0.6448, latency=0.3153 ms
app-1   | [INFO] IVF-PQ nprobe=16: recall=0.6611, latency=0.5752 ms
app-1   | [INFO] IVF-PQ nprobe=32: recall=0.6679, latency=1.0920 ms
app-1   | [INFO] IVF-PQ nprobe=64: recall=0.6718, latency=2.1239 ms
app-1   | [INFO] IVF-PQ nprobe=128: recall=0.6741, latency=4.1957 ms
app-1   | [INFO] HNSW ef_search=50: recall=0.9862, latency=6.0391 ms
app-1   | [INFO] HNSW ef_search=100: recall=0.9958, latency=6.5773 ms
app-1   | [INFO] HNSW ef_search=200: recall=0.9981, latency=7.8678 ms
app-1   | [INFO] HNSW ef_search=400: recall=0.9994, latency=9.5066 ms
app-1   | [INFO] FAZA 8: Generisanje grafika
app-1   | [INFO] Generisan: output/figures/tradeoff_curve.pdf
app-1   | [INFO] Generisan: output/figures/boxplot_latency.pdf
app-1   | [INFO] Generisan: output/figures/memory_comparison.pdf
app-1   | [INFO] Verifikacija hipoteza
app-1   | [INFO] H2 (memorija >=4x): flat=1464.8MB / ivfpq=35.8MB = 40.88x >= 4x = True
app-1   | [INFO] H2 (recall drop < 5pp): drop=0.3321 < 0.05 = False
app-1   | [INFO] H3 (budzet < 2GB): IVFPQ=35.8MB < 2048MB = True
app-1   | [INFO] H3 (IVF-PQ bolji tradeoff od HNSW): True (HNSW=2072.0MB ne staje u budzet)
app-1   | [INFO] Sve hipoteze potvrdjene: False
app-1   | [INFO] Ukupno vrijeme simulacije: 0h 17m 59s
app-1   | [INFO] Sve faze zavrsene. Grafici su spremljeni u output/figures/.
```