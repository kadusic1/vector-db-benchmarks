# Uvod

Kratak pregled potpoglavlja

## Motivacija
Zasto se vektorske baze danas bas mnogo koriste i zasto je ih potrebno
optimizovati tj. koji su problemi.

## Indeksi i kvantizacija u vektorskim bazama podataka
Spomenuti indekse FLAT sta znaci, HNSW, IVF, ScaNN, DiskANN

Spomenuti kvantizaciju kako je to bez kompresije, Binary, Scalar i Product
Quantiation

## Obim i ciljevi ovog rada
Opsti ciljevi

Hipoteze:

H1: HNSW indeks postiže statistički značajno nižu latenciju upita u poređenju s IVF indeksom na MS MARCO datasetu, uz zadržavanje recall@10 vrijednosti iznad 90%.

H2: Primjena Product Quantization tehnike smanjuje memorijski footprint vektorskog indeksa za najmanje 4x u poređenju s nekomprimiranim flat indeksom, uz smanjenje recall@10 manje od 5 procentnih poena.

H3: Kombinacija IVF i Product Quantization (IVF-PQ) pruža bolji recall-latency trade-off u poređenju sa standalone HNSW pristupom kada je memorijski budžet ograničen na ispod 2 GB.


# Teoretski okvir (Ovaj dio treba da udje u teoretske postavke, jednacine i slicno
, uvod je bio vise pregled ovoga, u ovom dijelu je dobro gdje moze imati i slike
primarno u svg ili pdf formatu tj. vektorskom formatu)

Kratak pregled potpoglavlja

## Vektorske baze podataka i embeddingsi

Kakve su to vektorske baze, sta su vektori, kako se vektori dobijaju, postavke jednacina,
za sta su korisni embeddingsi, spomenuti FAISS i Qdrant

SPomenuti i kreiranje embeddings i OPISATI DETALJNO modele koji se koriste za kreiranje
jednacine i slike, u ovom slucaju koristi se: all-mpnet-base-v2, sta znaci odabir modela.

## Vektorski indeksi

Objasniti i opisati kroz slike i jednacine: HNSW, IVF.

Samo spomenuti ScaNN i DiskANN oni nisu predmet ovog rada.

## Vektorska kvantizacija

Objasniti kroz jednacine i slike binary, scalar i product!

# Literatura

1. [Memory-Efficient Similarity Search at Billion-Scale: A Taxonomy and Analysis of Vector Compression Techniques](https://www.researchgate.net/publication/392469816_Memory-Efficient_Similarity_Search_at_Billion-Scale_A_Taxonomy_and_Analysis_of_Vector_Compression_Techniques)

2. [Optimization of Search Algorithms for High-Dimensional Data Spaces ](https://ijarcse.org/index.php/ijarcse/article/view/121/179)

3. [A Comprehensive Survey on Vector Database: Storage and Retrieval Technique, Challenge](https://arxiv.org/pdf/2310.11703)

# Metodologija
