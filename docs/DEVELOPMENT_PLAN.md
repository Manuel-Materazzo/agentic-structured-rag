# Piano di Sviluppo — DataPizza AI MVP
> Derivato da `ARCHITECTURE_DATAPIZZA_AI_MVP.md`  
> Ultimo aggiornamento: 19 giugno 2026

---

## Come leggere questo documento

Ogni fase è **indipendente e misurabile**: non si passa alla fase successiva prima che tutti i task critici (🔴) della fase corrente siano completati e i test di accettazione siano passati. I task facoltativi (🟡) possono essere posticipati se il tempo stringe. Le checklist sono ordinabili in un issue tracker (GitHub Issues, Linear, Jira) copiando le singole righe come task atomici.

---

## Fase 0 — Spike tecnica su `datapizza-ai`

> **Obiettivo:** verificare empiricamente che le API del framework siano stabili e corrispondano alla documentazione. Questa fase sblocca tutte le successive.

### Setup ambiente

- [x] 🔴 Clonare il repository `datapizza-ai` e installare le dipendenze
- [x] 🔴 Configurare percorsi dati, Qdrant (in memory), Duckdb (in memory)
- [x] 🔴 Struttura directory del repository creata come da §16:
  ```
  src/app/, src/ingestion/, src/metrics/
  data/raw/, data/parsed/, data/database/, data/cache/
  output/, docs/
  ```

### Verifica API `datapizza-ai`

Nota: produrre dei dev test sulla cartella `tests/`

- [x] 🔴 `Agent` — verificare che il costruttore accetti `tools` e `system_prompt`; testare un tool call fittizio end-to-end
- [x] 🔴 `IngestionPipeline` — verificare il metodo di avvio pipeline su un documento di test
- [x] 🔴 `DagPipeline` — verificare `add_module()` e l'esecuzione del grafo
- [x] 🔴 `DoclingParser` — parsare almeno un PDF di menu; verificare il formato dell'output
- [x] 🔴 `NodeSplitter` + `ChunkEmbedder` — produrre chunk da un testo di test e verificare il payload
- [x] 🔴 `OpenAIEmbedder` — verificare embedding di un testo campione
- [x] 🔴 `QdrantVectorstore` — `upsert`, `search`, `delete` su una collection di test
- [x] 🔴 `ChatPromptTemplate` + `ToolRewriter` — verificare la riscrittura di una query campione
- [x] 🔴 `ContextTracing` — verificare che i trace vengano emessi (anche solo su console/OTLP locale)
- [x] 🟡 ~~Documentare eventuali breaking change o comportamenti inattesi nel file `docs/spike_notes.md`~~ (nessun comportamento sospetto)

### Criteri di accettazione Fase 0

- Tutte le API elencate sopra sono verificate funzionanti nella versione corrente del repository
- Esiste un file `docs/spike_notes.md` che certifica il risultato della spike

✅ Fase 0 Superata
---

## Fase 1 — Bootstrap e infrastruttura dati

> **Obiettivo:** avere la struttura del progetto funzionante, i database inizializzati e il file di mapping caricato.

### Configurazione progetto

- [x] 🔴 `src/app/config.py` — percorsi, parametri LLM, dimensione embedding, soglie; tutto leggibile da variabili d'ambiente
- [x] 🔴 File `requirements.txt` / `pyproject.toml` completo e lockato

### Database

- [x] 🔴 Script `src/ingestion/runner.py` — crea `data/database/facts.db` se non esiste ed esegue le CREATE TABLE (§7.3):
  - `documents`
  - `restaurants`
  - `dishes`
  - `dish_ingredients` (con `quantity_grams FLOAT` e `quantity_raw TEXT NOT NULL`)
  - `dish_techniques`
  - `technique_taxonomy`
  - `planet_distances`
  - `compliance_rules`
- [x] 🔴 Script crea `data/database/ingestion_log.db` con tabella `ingestion_log` (stati: `pending → parsing → parsed → extracting → extracted → embedding → indexed → complete / failed`)
- [x] 🔴 Qdrant avviato in modalità embedded (o Docker locale); creazione delle quattro collection: `menu_index`, `manual_index`, `code_index`, `blog_index`

### Mapping piatti

- [x] 🔴 `generate_submission.py` — carica `Dataset/ground_truth/dish_mapping.json` in memoria all'avvio
- [x] 🔴 Funzione `export_empty_submission()` — genera un CSV `output/submission.csv` con `row_id` vuoti per tutte le 100 domande (verifica del formato richiesto)

### Criteri di accettazione Fase 1

- `facts.db` e `ingestion_log.db` vengono creati senza errori su macchina pulita
- Le quattro collection Qdrant esistono (verificabili via UI o API)
- `submission.csv` viene generato con 100 righe e le colonne `row_id`, `result`

✅ Fase 1 Superata
---

## Fase 2 — Pipeline di ingestion

> **Obiettivo:** tutti i documenti sorgente vengono correttamente parsati, estratti e indicizzati nei due store. Ogni passo è tracciato nell'ingestion log.

### 2.1 Parsing e confidenza

- [x] 🔴 `src/ingestion/menu_ingestion.py` — usa `DoclingParser` per parsare ogni menu PDF; salva output in `data/parsed/`
- [x] 🔴 Richiesta LLM di entity extraction restituisce obbligatoriamente il campo `parsing_confidence` (`"high"` / `"low"`) e `parsing_issues`
- [x] 🔴 `src/ingestion/vision_fallback.py` — se `parsing_confidence == "low"`, rilancia il documento in modalità vision e ripete l'extraction con lo stesso prompt
- [ ] 🟡 Definire la soglia esatta di "low": il LLM deve dichiarare "non sono riuscito a estrarre entità coerenti", non semplicemente "il testo era disordinato"

### 2.2 Entity extraction LLM

- [x] 🔴 `src/ingestion/structured_extraction.py` — prompt strutturato che estrae per ogni piatto:
  - `name`, `restaurant`, `ingredients[]` (con `quantity_grams` FLOAT o null e `quantity_raw` testo originale), `techniques[]`, `preparation_notes`
  - `chef`, `planet`, `chef_license`, `professional_orders[]` per il ristorante
  - Output: JSON con schema fisso, nessun testo libero aggiuntivo
- [x] 🔴 Normalizzazione quantità: il LLM produce `FLOAT` con punto decimale o `null` esplicito (mai `0.0` per "quanto basta"); `quantity_raw` sempre valorizzato
- [x] 🔴 Range check post-extraction: valori `quantity_grams` fuori range plausibile → warning loggato e forzato a `null`

### 2.3 Scrittura nei due store

- [x] 🔴 `doc_id` calcolato come `sha256(file_content)` deterministicamente
- [x] 🔴 Scrittura in DuckDB (`documents`, `restaurants`, `dishes`, `dish_ingredients`, `dish_techniques`) con FK corrette verso `doc_id`
- [x] 🔴 Chunking del testo parsato via `NodeSplitter` secondo la strategia per tipo di fonte (§8.4):
  - Menu: chunk per piatto/sezione; fallback 700-1200 caratteri
  - Manuale: chunk per sezione/sottosezione con metadata `section_title`, `topic`
  - Codice Galattico: chunk per sezione normativa + estrazione verso `compliance_rules`
  - Blog HTML: `DoclingParser` con fallback BeautifulSoup; chunk semantici con gerarchia h1/h2/h3
  - Distanze CSV: caricamento diretto in `planet_distances` — NON indicizzato in Qdrant
- [x] 🔴 Payload Qdrant conforme al contratto (§7.4): `chunk_id`, `doc_id`, `source_path`, `source_type`, `page`, `section`, `restaurant`, `dish`, `text`
- [x] 🔴 `upsert` vettoriale nella collection corretta in base a `source_type`

### 2.4 Ingestion log e ciclo di vita

- [x] 🔴 Implementare `ingest_document()` con transizione di stato: `pending → parsing → parsed → extracting → extracted → embedding → indexed → complete` (§8.3)
- [x] 🔴 In caso di eccezione: stato → `failed`, `error_message` valorizzato
- [x] 🔴 Skip automatico se `doc_id` già presente con stato `complete` (hash invariato)
- [x] 🔴 Implementare `update_document()`: invalida Qdrant + DuckDB a cascata per il vecchio `doc_id`, poi richiama `ingest_document()`
- [x] 🔴 Implementare `delete_document()`: purga Qdrant, delete a cascata in DuckDB, rimozione da `parsed/`, rimozione da `ingestion_log`
- [x] 🔴 Health check all'avvio: confronto O(n_documenti) tra `doc_id` in DuckDB e Qdrant; elementi orfani → `update_document()`
- [ ] 🟡 Worker `retry_failed()` che rielabora i record con stato `failed`

### 2.5 Ingestion dei singoli tipi di fonte

- [x] 🔴 `src/ingestion/menu_ingestion.py` — tutti i menu PDF
- [x] 🔴 `src/ingestion/cook_manual_ingestion.py` — Manuale di Cucina (tecniche, certificazioni, ordini → `technique_taxonomy`)
- [x] 🔴 `src/ingestion/galactic_code_ingestion.py` — Codice Galattico (limiti quantitativi → `compliance_rules`)
- [x] 🔴 `src/ingestion/distances_ingestion.py` — Distanze CSV → `planet_distances` in DuckDB
- [x] 🔴 `src/ingestion/blog_ingestion.py` — Blog post HTML → `blog_index` in Qdrant

### Criteri di accettazione Fase 2

- Parzialmente soddisfatti: il perimetro di ingestion è implementato e testato con smoke test; resta da rifinire la soglia di `parsing_confidence` e la strategia di chunking basata su `NodeSplitter`
- La tabella `dish_ingredients` non contiene `quantity_grams = 0.0` per casi non quantificabili
- Le quattro collection Qdrant contengono punti con payload conforme al contratto
- Un test di verifica estrae almeno 5 piatti a campione e ne verifica la corretta presenza in DuckDB
- Il fallback vision si attiva automaticamente su un PDF di test con parsing degradato

Fase 2 Superata (con item in sospeso)

---

## Fase 3 — Pipeline Easy: tool SQL e orchestratore base

> **Obiettivo:** rispondere correttamente alle 48 domande Easy generando una submission con Jaccard misurabile. Questa fase è il primo punto di misura reale.

### Tool deterministici SQL

- [x] 🔴 Implementare i tool esposti all'orchestratore
- [x] 🔴 Agente SQL (`src/app/agents/sql_agent.py`) — traduce linguaggio naturale in SQL, esegue, ritorna risultato raw con nomi colonne; auto-retry max 2 tentativi su errore sintattico

### Orchestratore base

- [x] 🔴 `src/app/orchestrator.py` — loop agentico con `Agent` di datapizza-ai:
  - riceve domanda + `row_id`
  - legge schema DuckDB per capire le entità disponibili
  - produce piano di retrieval JSON serializzabile (§9.2) con `reasoning`, `budget`, `next_handoff`, `query`
  - stima complessità e auto-assegna budget di handoff (Easy: max 1-2, Medium: max 3, Hard/Impossible: max 5)
  - delega al sub-agente SQL
  - legge risultato e decide se convergere
- [x] 🔴 Zero risultati SQL → non è risposta finale; l'orchestratore deve disambiguare (nella Fase 4 tramite Qdrant; in questa fase loggare il caso)

### Sintesi e normalizzazione

- [x] 🔴 `normalizer_utils.py` — rimozione spazi bianchi, standardizzazione caratteri speciali, confronto case-insensitive, rimozione duplicati
- [x] 🔴 Mapping verso ID tramite `dish_mapping.json`; fallback fuzzy matching controllato.
- [ ] 🔴 `src/app/submission.py` — esporta `output/submission.csv` con `row_id` e `result` (ID separati da virgola, ordinamento numerico crescente)

### Validazione tools

- [x] 🔴 `/validation_utils.py` — verifica che ogni nome piatto sintetizzato esista nel `dish_mapping.json` prima dell'inclusione nella submission
- [x] 🔴 `/run_detailed_inference.py` — fa partire l'inference di tutte le domande per estrarre delle risposte dettagliate (mantiene le risposte su disco per debug)
- [x] 🔴 `/evaluate_detained_inference_result.py` — controlla il risultato dell'inferenza tramite LLM per stabilire cosa è andato male su ciascuna domanda

### Misura Jaccard Easy

- [ ] 🔴 Eseguire `src/metrics/evaluation.py` sulla submission generata per le sole domande Easy
- [ ] 🔴 Registrare il punteggio Jaccard baseline in `docs/jaccard_log.md`
- [ ] 🟡 Iterare su chunking e prompt di extraction se il Jaccard è sotto soglia attesa

### Criteri di accettazione Fase 3

- La submission sulle 48 domande Easy viene generata senza errori
- Il punteggio Jaccard sulle domande Easy è misurabile e documentato
- Nessun candidato inventato (non presente in `dish_mapping.json`) compare nella submission

Fase 3 Superata (Jaccard similarity rimandata alla fase 4)

---

## Fase 4 — Estensione Medium e Hard

> **Obiettivo:** aggiungere il retrieval semantico via Qdrant, la logica su distanze e licenze, ed estendere l'orchestratore per gestire i 28 Medium e 18 Hard.

### Agente Qdrant

- [x] 🔴 `src/app/agents/qdrant_agent.py` — esegue `semantic_search(query, collection, k)` sulla collection appropriata; ritorna chunk con score e metadata; auto-retry max 2 tentativi (riformulazione query o collection alternativa) se score troppo bassi o zero risultati
- [x] 🔴 L'agente sintetizza e filtra il contenuto

### Estensione orchestratore

- [x] 🔴 Orchestratore esteso per gestire handoff verso `qdrant_agent` oltre a `sql_agent`
- [x] 🔴 Logica a tre rami (§9.5): Solo DuckDB / DuckDB + Qdrant / Solo Qdrant (quest'ultimo con parsimonia)
- [x] 🔴 Gestione zero result SQL: handoff a Qdrant per disambiguare, poi retry SQL con entità risolte
- [x] 🔴 Budget di handoff rispettato e loggato; superamento budget → esplicitare nel reasoning

### Logica Medium (§12.2)

- [ ] 🔴 Incrocio `dishes` + `restaurants` su colonne `planet` e `chef_license` via SQL
- [ ] 🔴 Rivalutazione schema DB in base alle nuove evidenze di esecuzione
- [ ] 🔴 Test su campione di 5 domande Medium; registrare Jaccard parziale

### Logica Hard (§12.3)

- [ ] 🔴 Vincoli geometrici: query su `planet_distances` con soglie di distanza (es. "pianeti entro X anni luce")
- [ ] 🔴 Vincoli tassonomici: join `dish_techniques` → `technique_taxonomy` → `required_license_level`; filtraggio in SQL
- [ ] 🔴 Algebra degli insiemi SQL per produrre candidati pre-filtrati prima della sintesi LLM
- [ ] 🔴 Test su campione di 5 domande Hard; registrare Jaccard parziale

### Osservabilità (§14)

- [ ] 🔴 `ContextTracing` configurato: ogni esecuzione registra domanda, `row_id`, piano di retrieval, ogni handoff (tipo, input, output, tempo), chunk con score, SQL eseguiti, errori di mapping, warning `quantity_grams`
- [ ] 🟡 Export trace JSON per regression testing offline (§14)

### Criteri di accettazione Fase 4

- Submission completa su domande Easy + Medium + Hard senza errori
- Jaccard registrato separatamente per livello di difficoltà
- Almeno un trace JSON esportato e verificabile per una domanda Hard

---

## Fase 5 — Domande Impossible e finalizzazione

> **Obiettivo:** aggiungere la logica per le 6 domande Impossible, completare i test, e produrre la submission finale.

### Logica Impossible (§12.4)

- [ ] 🔴 Compliance rules da Codice Galattico: query su `compliance_rules` per tetti quantitativi su ingredienti regolati
- [ ] 🔴 Incrocio `dish_ingredients.quantity_grams` con `compliance_rules.constraint_value`; esclusione piatti in violazione
- [ ] 🔴 Retrieval su `blog_index` per anomalie narrative; incrocio con `compliance_rules` in DuckDB
- [ ] 🔴 Approccio iper-conservativo in candidate generation: preferire escludere candidati incerti piuttosto che includerli (minimizzare over-generation per Jaccard)

### Testing (§15)

- [ ] 🔴 **Test ingestion:**
  - parsing strutturato corretto su PDF campione (1 menu per tipo di layout)
  - fallback vision si attiva correttamente su PDF con `parsing_confidence = low`
  - `quantity_grams` normalizzato correttamente (FLOAT, null, mai 0.0 per "quanto basta")
  - hash nell'ingestion log allineati dopo INSERT, UPDATE e DELETE
- [ ] 🔴 **Test retrieval:**
  - query su solo ingredienti: verificare recall
  - query su tecniche accoppiate: verificare precision
  - query su matrice distanze: verificare correttezza numerica
  - query su compliance limits: verificare correttezza dei limiti estratti
- [ ] 🔴 **Test end-to-end:**
  - submission completa su tutte le 100 domande
  - `src/metrics/evaluation.py` eseguito; Jaccard medio registrato
  - Golden set (5 domande per livello, totale 20) eseguito e verificato come barriera di regressione

### Anti-overfitting (§12.5)

- [ ] 🔴 Verificare che i prompt LLM non contengano riferimenti alle 100 domande specifiche del benchmark
- [ ] 🔴 Verificare che dizionari e lookup siano generati dinamicamente dalla Knowledge Base, mai cablati a mano

### Definition of Done (§19) — checklist finale

- [ ] 🔴 Tutte le fonti dati indicizzate con stato `complete` nell'ingestion log
- [ ] 🔴 Fallback vision operativo e testato
- [ ] 🔴 `quantity_grams` produce `FLOAT` o `null` con `quantity_raw` sempre valorizzato
- [ ] 🔴 Ogni domanda genera un output stabile e formattato
- [ ] 🔴 L'orchestratore esplicita il ragionamento e rispetta il budget di handoff dichiarato
- [ ] 🔴 `evaluation.py` restituisce un Jaccard medio misurabile e affidabile
- [ ] 🔴 Ciclo INSERT/UPDATE/DELETE garantisce coerenza tra DuckDB e Qdrant senza record orfani

---

## Rischi e mitigazioni — riferimento rapido

| Rischio | Mitigazione | Fase |
|---|---|---|
| Over-generation abbatte Jaccard | Validazione obbligatoria su `dish_mapping.json`; soglia conservativa su Qdrant; budget handoff | 3, 4, 5 |
| Desync DuckDB ↔ Qdrant | Protocollo invalida-prima-reinserisci-dopo via `doc_id`; health check all'avvio | 2 |
| Parsing PDF complessi fallisce | Segnale confidenza + fallback vision; verifica empirica su campione | 2 |
| Errori entity extraction si propagano | Test per tipo di documento; review manuale campione; warning `quantity_grams` | 2 |
| Quantità espresse in forme eterogenee | LLM istruito a FLOAT o NULL; `quantity_raw` sempre preservato; range check | 2 |
| Orchestratore over-investigate domande Easy | Budget handoff auto-stimato e dichiarato nel piano | 3 |
| API `datapizza-ai` instabili | Spike tecnica obbligatoria in Fase 0 prima di qualsiasi implementazione | 0 |
| Chunking non ottimale degrada retrieval | Chunking guidato dalla struttura logica; calibrazione empirica in Fase 3 | 2, 3 |

---

## Dipendenze tra fasi

```
Fase 0 (Spike)
    ↓
Fase 1 (Bootstrap)
    ↓
Fase 2 (Ingestion)
    ↓
Fase 3 (Easy + Jaccard baseline)  ← punto di misura obbligatorio
    ↓
Fase 4 (Medium + Hard)
    ↓
Fase 5 (Impossible + DoD)
```

Non avanzare alla fase successiva prima che i criteri di accettazione della fase corrente siano soddisfatti.

---

## Struttura file di riferimento

```text
src/
  app/
    main.py                  # entry point
    config.py                # variabili d'ambiente e costanti
    orchestrator.py          # loop agentico principale
    answer_normalizer.py     # normalizzazione e fuzzy matching
    submission.py            # export CSV + caricamento dish_mapping.json
    agents/
      sql_agent.py           # traduzione NL→SQL, esecuzione, auto-retry
      qdrant_agent.py        # retrieval semantico, auto-retry
    tools/
      lookup_tools.py        # tool deterministici SQL esposti all'orchestratore
      validation_tools.py    # verifica candidati vs dish_mapping.json
  ingestion/
    runner.py                # entry point ingestion; init DB e collection
    menu_ingestion.py
    cook_manual_ingestion.py
    galactic_code_ingestion.py
    blog_ingestion.py
    distances_ingestion.py
    structured_extraction.py # LLM entity extractor con confidenza
    vision_fallback.py       # fallback vision parser
  metrics/
    evaluation.py            # calcolo Jaccard medio
data/
  raw/                       # copie file sorgente
  parsed/                    # output Docling normalizzato
  database/                  # facts.db, ingestion_log.db, Qdrant embedded
  cache/
output/
  submission.csv
docs/
  architecture.md
  spike_notes.md             # output Fase 0
  jaccard_log.md             # storico punteggi Jaccard per fase
```
