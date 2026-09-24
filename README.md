# Classificatore di parole con un Transformer encoder

Questo progetto contiene un Transformer encoder didattico scritto interamente
in C++. Il programma classifica parole ASCII in due classi: parola esistente
oppure parola non esistente.

Ogni carattere viene convertito in un embedding apprendibile di dimensione 10.
Il programma include sia il forward pass sia un training supervisionato con
backpropagation.

L'esempio è pensato per capire i passaggi fondamentali del modello, non per sostituire framework come PyTorch o TensorFlow.

## Requisiti

- un compilatore C++ con supporto a C++17;
- PowerShell, Prompt dei comandi o un terminale equivalente.

Non sono necessarie librerie esterne.

## Compilazione

Dalla cartella del progetto eseguire:

```powershell
g++ -std=c++17 -Wall -Wextra -pedantic main.cpp -o transformer_encoder.exe
```

Su Windows, il comando richiede che `g++` sia installato e disponibile nel `PATH`, per esempio tramite MinGW-w64.

## Esecuzione

Il programma ha due modalità separate: `train` per addestrare e salvare i pesi, `predict` per caricare i pesi e classificare senza fare training.

### Training

Per addestrare il modello da zero:

```powershell
.\transformer_encoder.exe train
```

È possibile passare uno o più file di training. Il programma concatena tutti
gli esempi nell'ordine indicato:

```powershell
.\transformer_encoder.exe train datasets/morphit_ascii_data.txt datasets/morphit_ascii_negative_data.txt
```

Se non vengono indicati file, viene usato `datasets/training_set.txt`.
Il numero di epoche predefinito è `1000`, ma si può modificare:

```powershell
.\transformer_encoder.exe train datasets/training_set.txt --epochs 3
```

È possibile interrompere automaticamente il training quando la loss media
dell'epoca scende sotto una soglia:

```powershell
.\transformer_encoder.exe train datasets/training_set.txt --epochs 100 --loss-threshold 0.01
```

Per usare una validation loss e fermare il training dopo `5` epoche senza
miglioramenti:

```powershell
.\transformer_encoder.exe train datasets/training_set.txt --validation datasets/validation_set.txt --epochs 100 --patience 5
```

L'opzione `--validation` può essere ripetuta per concatenare più file di
validation. Dopo ogni epoca il programma calcola la validation loss senza
aggiornare i pesi, salva in memoria i pesi migliori e ripristina quelli alla
fine del training.

Per uno smoke test si può limitare il numero di esempi letti da ogni file:

```powershell
.\transformer_encoder.exe train --limit 10 --validation datasets/validation_set.txt --epochs 2 --patience 1
```

Il programma stamperà il numero effettivo di esempi caricati. `--limit 10`
legge i primi 10 esempi di ciascun file, senza creare dataset temporanei. La
loss e l'accuratezza ottenute in questo modo servono solo a verificare che il
codice funzioni, non a valutare la qualità del modello.

Per continuare un training precedente dall'ultima epoca completata:

```powershell
.\transformer_encoder.exe train datasets/training_set.txt --validation datasets/validation_set.txt --epochs 1 --patience 5 --resume
```

`--resume` carica `models/transformer_model_latest.txt`. Al termine di ogni epoca il
programma salva sempre quel checkpoint; `models/transformer_model.txt` contiene invece
il modello con la validation loss migliore. I checkpoint includono anche
metadati: epoca completata, epoca migliore, training loss e validation loss.

Il programma stampa l'inizio e la fine di ogni epoca, il tempo impiegato e la
loss. Alla fine salva il modello in `models/transformer_model.txt`:

```text
Inizio Epoca #1.
Fine Epoca #1. Tempo impiegato 00:42. Training loss: ...
...
Modello migliore salvato in models/transformer_model.txt
```

### Predict

Dopo aver eseguito il training, per usare il modello senza riaddestrarlo:

```powershell
.\transformer_encoder.exe predict
```

La modalità `predict` carica `models/transformer_model.txt`, esegue soltanto il forward pass e stampa le predizioni:

```text
Target: 0, predizione: 0
Target: 1, predizione: 1
Accuratezza: 4/4
```

I valori esatti della loss possono cambiare se si modificano le inizializzazioni o il dataset.

## Cosa succede durante una run

Le modalità sono indipendenti e usano file diversi:

1. `train` legge uno o più file indicati sulla riga di comando, concatena i
   dataset, inizializza i pesi, addestra encoder e classificatore per `1000`
   epoche predefinite (oppure per il valore passato con `--epochs`) e salva il
   modello; se viene indicata una validation, usa anche `--patience` per
   l'early stopping. Con `--resume` carica l'ultimo checkpoint invece di
   inizializzare pesi nuovi;
2. `predict` legge `models/transformer_model.txt` e `datasets/test_set.txt`, crea le rappresentazioni con il forward pass e classifica gli esempi.

Il comando:

```powershell
.\transformer_encoder.exe train
```

non esegue predizioni alla fine del training. Per classificarli bisogna eseguire separatamente:

```powershell
.\transformer_encoder.exe predict
```

Ogni nuova esecuzione di `train` riparte da pesi iniziali nuovi, a meno che non
si specifichi `--resume`. Il training salva l'ultima epoca in
`models/transformer_model_latest.txt` e il miglior modello secondo la
validation loss in `models/transformer_model.txt`. Una nuova esecuzione di
`predict`, invece, carica `models/transformer_model.txt` e non modifica i
pesi.

Se si esegue `predict` prima di `train`, il programma mostra un errore perché `models/transformer_model.txt` non esiste ancora. Il dataset di test è separato da quello di training, quindi l'accuratezza misura il comportamento su esempi non usati per aggiornare i pesi.

I comandi disponibili sono:

```text
programma train [dataset1 dataset2 ...] [--validation file] [--epochs N]
                 [--patience N] [--loss-threshold X] [--limit N] [--resume]
programma predict [dataset] [--limit N]
```

Il file dei pesi è testuale e contiene la tabella degli embedding ASCII,
matrici, vettori `gamma`/`beta`, bias e parametri del classificatore.

## Formato dei dataset

I file `datasets/training_set.txt`, `datasets/validation_set.txt` e
`datasets/test_set.txt` usano lo stesso formato testuale. La
prima riga contiene il numero di esempi. Ogni esempio è composto da label e
parola:

```text
label parola
```

Per esempio:

```text
1
1 casa
0 qzmt
```

La label `1` indica una parola esistente nel piccolo dataset didattico; la
label `0` indica una parola inventata. Il programma accetta caratteri ASCII e
può gestire parole di lunghezza diversa. Non vengono usati padding o batch: un
esempio alla volta viene convertito in una matrice `lunghezza_parola × 10`.

Per aggiungere esempi di training, modificare il dataset appropriato in
`datasets/`. Le label devono essere `0` o
`1` e le parole devono contenere soltanto caratteri ASCII.

### Dataset pulito da Morph-it!

Il repository contiene anche `datasets/morphit_ascii_data.txt`, ottenuto dal file
Morph-it! originale. Il file contiene 382.520 forme distinte, normalizzate in
minuscolo e filtrate per conservare soltanto parole composte dalle lettere
ASCII `a-z`:

```text
382520
1 abbadessa
1 abbandonare
...
```

Il file è quindi compatibile con il parser del programma, ma contiene soltanto
esempi positivi: da solo non è ancora un dataset binario equilibrato per il
training. Può essere passato insieme al file negativo nella modalità `train`.

È presente anche `datasets/morphit_ascii_negative_data.txt`, con 382.520 esempi
negativi. Ogni parola nasce dalla forma positiva corrispondente tramite da 1 a
3 operazioni casuali: sostituzione oppure aggiunta di un carattere. Le parole
generate sono uniche e non coincidono con nessuna forma positiva del database.
Il file usa label `0`.

Per rigenerare il file a partire dal database originale:

```powershell
python3 tools/generate_morphit_dataset.py \
  /percorso/morph-it_048.txt \
  --train-positive 1000 \
  --test-positive 250 \
  --cleaned-output datasets/morphit_ascii_data.txt \
  --negative-output datasets/morphit_ascii_negative_data.txt
```

Lo script legge il formato Morph-it! `forma`, `lemma`, `categoria`, separato da
tabulazioni. Scarta accenti, punteggiatura, cifre e forme non ASCII. Il numero
massimo di modifiche delle parole negative si può cambiare con
`--max-changes`.

Per creare direttamente i tre dataset bilanciati, con divisione `80% / 10% /
10%`, usare:

```powershell
python3 tools/generate_morphit_dataset.py \
  /percorso/morph-it_048.txt \
  --split-training-output datasets/training_set.txt \
  --split-validation-output datasets/validation_set.txt \
  --split-test-output datasets/test_set.txt
```

I tre file risultanti contengono rispettivamente:

- `training_set.txt`: 612.032 esempi, metà positivi e metà negativi;
- `validation_set.txt`: 76.504 esempi, metà positivi e metà negativi;
- `test_set.txt`: 76.504 esempi, metà positivi e metà negativi.

Le parole positive e negative vengono prima abbinate, poi le coppie vengono
mescolate e distribuite nei tre file. In questo modo ogni split conserva lo
stesso rapporto tra le due label.

## Cosa fa il programma

Il programma classifica parole molto piccole appartenenti a due classi:

- classe `0`: parola non presente negli esempi positivi;
- classe `1`: parola presente negli esempi positivi.

La tabella degli embedding ha forma `128 × 10`: una riga per ogni carattere
ASCII e dieci coordinate apprendibili per carattere. Gli embedding vengono
inizializzati casualmente con Xavier e aggiornati durante il training.

Il flusso del modello è:

```text
parola ASCII
    -> caratteri ASCII
    -> character embedding 128 x 10
    -> positional encoding
    -> self-attention singola
    -> residual connection
    -> layer normalization affine
    -> rete feed-forward con ReLU
    -> residual connection
    -> layer normalization affine
    -> mean pooling
    -> classificatore lineare
    -> softmax
    -> cross-entropy
    -> backpropagation
```

## Componenti addestrabili

Il training aggiorna tutti i principali parametri del modello:

- tabella degli embedding ASCII `128 x 10`;
- `Wq`, `Wk`, `Wv`: proiezioni di query, key e value;
- `W1`, `W2`: pesi della rete feed-forward;
- `gammaAttention`, `betaAttention`: parametri della prima layer normalization;
- `gammaOutput`, `betaOutput`: parametri della seconda layer normalization;
- pesi e bias del classificatore finale.

I pesi delle matrici vengono inizializzati con Xavier initialization. `gamma` parte da `1`, mentre `beta` parte da `0`.

## Self-attention

L'esempio usa una sola testa di attenzione per mantenere il codice semplice. La formula è:

$$
Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Le proiezioni sono calcolate con:

```text
Q = input * Wq
K = input * Wk
V = input * Wv
```

Non viene usata la multi-head attention.

## Layer normalization

Per ogni token viene calcolata la normalizzazione delle sue feature. In seguito si applica la trasformazione affine:

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

Durante il training anche `gamma` e `beta` ricevono gradienti e vengono aggiornati con la discesa del gradiente.

## Training

Per ogni parola il programma esegue:

1. conversione dei caratteri nelle righe della tabella di embedding;
2. calcolo del forward pass dell'encoder;
3. mean pooling dell'output dell'encoder;
4. calcolo dei logits del classificatore;
5. softmax e cross-entropy;
6. calcolo dei gradienti nel backward pass;
7. aggiornamento di encoder, classificatore e embedding dei caratteri usati.

La struttura `EncoderCache` conserva i valori intermedi del forward, come `Q`, `K`, `V`, le probabilità dell'attenzione e le attivazioni della rete feed-forward. Questi valori servono per calcolare i gradienti senza ripetere il forward.

I parametri principali del training sono definiti in `main`:

```cpp
const double learningRate = 0.01;
const std::size_t defaultEpochs = 1000;
```

Per un training più lento o più veloce si può modificare `learningRate`. Il
numero di epoche si passa alla riga di comando con `--epochs N`; se l'opzione
non viene specificata, viene usato `defaultEpochs`. Con
`--loss-threshold X` il training si interrompe dopo la prima epoca la cui loss
media è inferiore a `X`. Con `--patience N`, se è presente una validation, il
training si interrompe dopo `N` epoche consecutive senza miglioramenti della
validation loss.

## File del progetto

- `main.cpp`: implementazione degli embedding ASCII, dell'encoder, del backward pass, del classificatore e del training;
- `theory.md`: spiegazione teorica di Xavier initialization, loss, cross-entropy, pooling, cache, layer normalization, `gamma` e `beta`;
- `datasets/training_set.txt`: dataset Morph-it! completo per il training;
- `datasets/validation_set.txt`: dataset Morph-it! per validation loss ed early stopping;
- `datasets/test_set.txt`: dataset Morph-it! per la valutazione finale;
- per gli smoke test si usa `--limit` sui dataset principali, senza creare dataset aggiuntivi;
- `models/transformer_model.txt`: miglior modello salvato da `train` e caricato da `predict`, con metadati della validation loss;
- `models/transformer_model_latest.txt`: pesi dell'ultima epoca completata, usati da `train --resume`;
- `transformer_encoder.exe`: eseguibile generato dalla compilazione, se si usa il comando mostrato sopra.

## Limiti dell'esempio

Questo programma è intenzionalmente minimale:

- usa una sola testa di attenzione;
- usa pesi inizializzati casualmente ma con generatori deterministici;
- usa una tabella di embedding per i 128 caratteri ASCII;
- non usa un tokenizer linguistico: ogni carattere ASCII è un token;
- non crea automaticamente una separazione tra training, validation e test:
  i file di validation devono essere indicati esplicitamente con
  `--validation`;
- non implementa dropout, maschere di attenzione o batch training;
- usa matrici costruite con `std::vector`, quindi non è ottimizzato per sequenze grandi;
- il training è pensato per essere eseguito su PC, non direttamente su Arduino.

Per usare il modello su Arduino, un possibile passo successivo sarebbe addestrare i pesi sul PC, salvarli in un formato compatibile e implementare sull'Arduino soltanto il forward pass.
