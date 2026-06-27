## Descrizione della sfida 🪐

_(Disclaimer: questa sezione iniziale è puramente flavour ed era stata usata per la sfida "Hackapizza". Leggere la sezione successiva per avere tutto il necessario per completare il test tecnico)_

**Benvenuti** nel **Ciclo Cosmico 789**, dove l'umanità ha superato non solo i confini del proprio sistema solare, ma anche quelli delle dimensioni conosciute. In questo vasto intreccio di realtà e culture, la gastronomia si è evoluta in un'arte che trascende spazio e tempo. 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6840884%2Fd4cd3a9d619dec67942e5344dcacf9e4%2F9gw32h.gif?generation=1737047022355670&alt=media)

Ristoranti di ogni tipo arricchiscono il tessuto stesso del multiverso: dai sushi bar di **Pandora** che servono prelibati sashimi di **Magikarp** e ravioli al **Vaporeon**, alle taverne di **Tatooine** dove l’**Erba Pipa** viene utilizzata per insaporire piatti prelibati, fino ai moderni locali dove lo **Slurm** compone salse dai sapori contrastanti - l'universo gastronomico è vasto e pieno di sorprese.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6840884%2F888315aac2d2bdd249e8df8fc79f8043%2Fimage.png?generation=1737046855158236&alt=media)

L'espansione galattica ha portato con sé nuove responsabilità. La **Federazione Galattica** monitora attentamente ogni ingrediente, tecnica di preparazione e certificazione necessaria per garantire che il cibo servito sia sicuro per tutte le specie senzienti. Gli **chef** devono destreggiarsi tra regolamenti complessi, gestire ingredienti esotici che esistono simultaneamente in più stati quantici e rispettare le restrizioni alimentari di centinaia di specie provenienti da ogni angolo del **multiverso**.

Nel cuore pulsante di questo arcipelago cosmico di sapori, si erge un elemento di proporzioni titaniche, un'entità che trascende la mera materialità culinaria: la **Pizza Cosmica**. Si narra che la sua mozzarella sia stata ricavata dalla **Via Lattea** stessa e che, per cuocerla, sia stato necessario il calore di tre soli. Nessuno conosce le sue origini e culti religiosi hanno fondato la loro fede attorno al suo mistero.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6840884%2F0c07b3e6f34ac48b9bb627387ce71531%2FTesto%20del%20paragrafo%20(1).png?generation=1737047186767633&alt=media)

***Che la forza sia con voi.***

## Test Tecnico 

### Specifiche tecniche 💻

Ti sarà richiesto di creare una repository Github che contenga il codice per risolvere in maniera parziale o totale questo test.

Il sistema GenAI che creerai dovrà essere in grado di rispondere alle domande presenti in questo [csv](./Dataset/domande.csv).

Le domande sono in linguaggio naturale e hanno come risposta univoca una lista di piatti. Ad esempio, la prima domanda "Quali sono i piatti che includono le Chocobo Wings come ingrediente?", ha come risposta "Galassia di Sapori: Il Viaggio Senza Tempo", mentre la domanda 10 "Quali piatti eterei sono preparati usando sia la Cottura Olografica Quantum Fluttuante che la Decostruzione Interdimensionale Lovecraftiana?" ha come risposta i piatti "Risotto dei Multiversi", "La Mucca Che Stordisce l'Universo" e "Sogni di Abisso Cosmico".

Le domande sono ordinate per difficoltà e per tipologia. Per la precisione:
- le domande di difficoltà **"Easy"** riguardano solo gli Ingredienti e le Tecniche, pertanto bastano solo i [Menu](./Dataset/knowledge_base/menu/) di ciascun ristorante
- le domande di difficoltà **"Medium"** riguardano anche le Licenze e i Pianeti. Nei [Menu](./Dataset/knowledge_base/menu/) sono descritti il livello di Licenza di ogni Chef e il Pianeta su cui si trova il ristorante. Sebbene non necessario, all'interno del [`Manuale di Cucina.pdf`](./Dataset/knowledge_base/misc/Manuale%20di%20Cucina.pdf) e [`Codice Galattico.pdf`](./Dataset/knowledge_base/codice_galattico/Codice%20Galattico.pdf) vi è una descrizione di come funzionano le licenze.
- le domande di difficoltà **"Hard"** riguardano le distanze tra pianeti, i tipi di cottura/preparazione e la licenza necessaria per la preparazione (ogni piatto necessita di certe tecniche e ogni tecnica necessita di certe licenze). Nel [`Distanze.csv`](./Dataset/knowledge_base/misc/Distanze.csv) c'è la tabella delle distanze tra pianeti. Il pdf [Manuale di Cucina.pdf](./Dataset/knowledge_base/misc/Manuale%20di%20Cucina.pdf) contiene le ultime due informazioni.
- le domande di difficoltà **"Impossible"** riguardano piccoli dettagli che si trovano all'interno di [`Codice Galattico.pdf`](./Dataset/knowledge_base/codice_galattico/Codice%20Galattico.pdf) e [`Blog post`](./Dataset/knowledge_base/blogpost/)

> [!WARNING]
> ⚠️⚠️⚠️**IMPORTANTE**⚠️⚠️⚠️: Se hai poco tempo, puoi tranquillamente fermarti SOLO alle domande di difficoltà "Easy". Decidi tu se vuoi migliorare la tua soluzione esistente o cercare di trovare soluzioni per domande più difficili. Un sistema capace di rispondere alle domande "Easy" è già un buon risultato.

### Descrizione Knowledge Base 📋

Dentro la cartella [knowledge_base](./Dataset/knowledge_base), ci sono tutti i file necessari per l'applicativo GenAI per rispondere alle domande.   

All'interno troverai i seguenti file e cartelle:

- [`Menu (30 ristoranti)`](./Dataset/knowledge_base/menu/)
    
    - Documenti in pdf contenenti i menù di 30 ristoranti differenti
    - I menu descrivono in linguaggio naturale il ristorante, riportando il nome dello Chef, il nome del ristorante, (laddove presente) il pianeta su cui c'è il ristorante e le licenze culinarie che ha lo chef
    - Ogni menu contiene 10 piatti
    - Ogni piatto contiene gli ingredienti usati e le tecniche di preparazione
    - Alcuni menu possiedono anche una descrizione in linguaggio naturale della preparazione
    - Laddove vi siano certi *ordini professionali*, i menu lo citano

- [`Manuale di Cucina.pdf`](./Dataset/knowledge_base/misc/Manuale%20di%20Cucina.pdf)
    
    Manuale di cucina che include:
    
    - L’elenco e la descrizione delle certificazione che uno chef può acquisire
    - L’elenco degli ordini professionali gastronomici a cui uno chef può aderire
    - L’elenco e la descrizione delle tecniche culinarie di preparazione esistenti
    - \[Hint\] La maggior parte del documento descrive nel dettaglio le tecniche disponibili. Ci sono circa 10 macrocategorie di tecniche culinarie dove ciascuna di esse comprende circa 5 tecniche. Alcuni utenti potrebbero richiedere piatti con una specifica macrocategoria di tecnica o una specifica tecnica.
    - \[Hint\] La maggior parte del testo è flavour e non serve per rispondere alle domande.
    - \[Hint\] Gli ordini professionali sono perlopiù usati da alcuni utenti che esprimono una preferenza verso una specifica tecnica. Questa tecnica in genere è riportata nei menu attraverso l'uso di emoji + glossario.

- [`Distanze.csv`](./Dataset/knowledge_base/misc/Distanze.csv)
    Un csv che contiene la matrice delle distanze in anni luce tra i pianeti su cui si trovano i diversi ristoranti.    
    \[Hint\] Alcune domande fanno richiesta di piatti entro una certa distanza. Ogni ristorante (eccetto uno) si trova su un pianeta.

- [`Codice Galattico.pdf`](./Dataset/knowledge_base/codice_galattico/Codice%20Galattico.pdf)
    
    Un documento legislativo contenente:
    
    - Limiti quantitativi applicati all'utilizzo di alcuni ingredienti nella preparazione dei piatti
    - \[Hint\] Alcune domande richiedono piatti che rispettino questi limiti. Per rispondere, il sistema deve: (1) identificare se il piatto contiene ingredienti regolamentati, e (2) verificare che le quantità usate non superino i limiti previsti dal Codice Galattico.
    - Vincoli relativi alle certificazioni che gli chef hanno bisogno di acquisire per poter utilizzare specifiche tecniche di preparazione dei piatti
    - \[Hint\] Alcuni utenti potrebbero chiedere che lo chef che prepara il piatto abbia le certificazioni a norma per cucinare tale piatto, pertanto è necessario controllare per ogni tecnica se lo chef ha la certificazione al livello corretto
    - \[Hint\] Le informazioni contenute in questo documento sono le più difficili da estrarre e rielaborare dell'intero test tecnico. Tuttavia, hanno impatto solo sulle ultime 4 domande del [csv](./Dataset/domande.csv).

- [`Blog post`](./Dataset/knowledge_base/blogpost/)

    - Pagine HTML che contengono informazioni supplementari su alcuni ristoranti
    - \[Hint\] Sono necessari solo per un numero limitato di domande, da usare congiuntamente con il Codice Galattico.pdf


### Evaluation

Per supportare lo sviluppo e la verifica del tuo sistema, nella cartella [dataset/ground_truth](./Dataset/ground_truth) troverai i file necessari per l'evaluation.

**Attenzione**: la ground truth non deve essere utilizzata dal sistema GenAI per generare le risposte, ma serve esclusivamente per valutare le performance. Il dataset è suddiviso in *public* e *private* (vedi colonna "Usage" in [`ground_truth_mapped.csv`](./Dataset/ground_truth/ground_truth_mapped.csv)) nel caso tu voglia suddividere test e validation.

L'evaluation misura la correttezza delle risposte confrontando i piatti restituiti dal tuo sistema con quelli attesi.
La metrica utilizzata è la **Jaccard Similarity**, calcolata per ogni domanda come l'intersezione diviso l'unione degli ID dei piatti.
Il punteggio finale è la **media** della Jaccard Similarity su tutte le domande, moltiplicata per 100.

#### Formato della Submission

Il tuo sistema dovrà produrre un file CSV contenente le risposte per tutte le domande presenti in [domande.csv](./Dataset/domande.csv).
Il file deve avere le colonne `row_id` e `result`:

```csv
row_id,result
1,"23,122"
2,"12"
3,"11,87"
4,"34,43"
5,"112"
6,"56"
7,"99"
8,"102,103"
9,"11"
10,"11,34"
...
```

**Dettagli dei campi:**
- `row_id`: l'ID progressivo della domanda (corrispondente alla riga nel file [domande.csv](./Dataset/domande.csv)), incrementale a partire da 1.
- `result`: una stringa contenente gli ID dei piatti identificati, separati da virgola.
    - **Nota**: il campo non può essere vuoto. Si assume che esista sempre almeno un piatto che soddisfi la query.
    - **Mapping**: per ottenere gli ID corretti, associa i nomi dei piatti trovati agli ID corrispondenti utilizzando il file [dish_mapping.json](./Dataset/ground_truth/dish_mapping.json).

#### Esempio

**Domanda**: "Vorrei assaggiare l'Erba Pipa. In quali piatti la posso trovare?"

Immaginiamo che il tuo sistema ritorni come risposta:

```json
["Risotto all'Erba Pipa", "Insalata Galattica"]
```

Se il file `dish_mapping.json` contiene:
```
{
    ...
    "Risotto all'Erba Pipa": 1,
    ...
    "Insalata Galattica": 5,
    ...
}
```
La risposta attesa nel CSV per questa domanda sarà `"1,5"`.
Se questa è la domanda `1`, allora:

```csv
row_id,result
1,"1,5"
...
```

#### Eseguire l'Evaluation

Una volta generato il file CSV con le tue risposte, puoi calcolare il punteggio eseguendo lo script fornito:

```bash
python src/evaluation.py --submission path/to/your_submission.csv
```

Lo script mostrerà il **Jaccard similarity score** medio complessivo.

