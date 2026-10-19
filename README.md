# 🔍 Moteur de recherche sémantique distribué

Moteur de recherche qui retrouve des documents selon leur **sens**, avec comparaison de plusieurs stratégies d'encodage parallèle.

## Fonctionnalités

- Chargement de documents `.txt` et `.pdf`
- Encodage sémantique multilingue (`paraphrase-multilingual-MiniLM-L12-v2`)
- Recherche par similarité cosinus (top 5) avec interface **Streamlit**
- Trois stratégies d'encodage parallèle comparées : **joblib**, **multiprocessing**, **MPI**
- Sauvegarde des résultats de recherche dans `search_results.txt`

## Structure du projet

```
semantic-search/
├── app.py                  # Interface Streamlit
├── benchmark.py            # Compare séquentiel, joblib et multiprocessing
├── requirements.txt
├── data/corpus_txt/        # Corpus d'exemple (5 textes)
└── src/
    ├── loader.py           # Chargement des .txt / .pdf
    ├── search_engine.py    # Encodage, recherche, sauvegarde des résultats
    └── encoders/
        ├── joblib_encoder.py
        ├── multiprocessing_encoder.py
        └── mpi_encoder.py
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**MPI (optionnel, Windows)** : installer [Microsoft MPI v10.0](https://www.microsoft.com/en-us/download/details.aspx?id=57467), puis `pip install mpi4py`.

## Utilisation

Interface de recherche :

```bash
streamlit run app.py
```

Comparaison des stratégies d'encodage (séquentiel, joblib, multiprocessing) :

```bash
python benchmark.py
```

Encodage distribué avec MPI (2 processus) :

```bash
mpiexec -n 2 python -m src.encoders.mpi_encoder
```

Exemple de requête : `Impacts de l'intelligence artificielle sur le travail humain` → `ai_revolution.txt` arrive en tête.

## Résultats de performance

| Stratégie | Temps | Remarque |
|---|---|---|
| Séquentiel | ~0,1 s | modèle déjà chargé |
| MPI (2 processus) | ~0,5 s | mesuré séparément, modèle déjà chargé |
| joblib (une tâche par document) | ~15 s | démarrage des processus et copie du modèle |
| multiprocessing | ~18 s | chaque processus recharge le modèle |

**Conclusion** : sur un corpus aussi petit (5 documents), le parallélisme est **plus lent** que l'encodage séquentiel, car le coût de coordination dépasse le gain de calcul. Il devient intéressant sur un grand corpus ou sur plusieurs machines (MPI).

## Limites et pistes d'amélioration

- Benchmark sur un corpus de plusieurs centaines ou milliers de documents
- Découpage des longs documents en passages (le modèle tronque les textes longs)
- Index vectoriel (FAISS) pour de grands volumes
- Détection automatique des changements du corpus.
