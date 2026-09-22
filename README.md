# Esempio minimale di Transformer encoder

Questo progetto contiene un Transformer encoder didattico scritto interamente in C++. Il programma include sia il forward pass sia un training supervisionato con backpropagation.

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

Il programma stampa periodicamente la loss e salva il modello in `transformer_model.txt`:

```text
Epoca 1, loss: ...
Epoca 201, loss: ...
...
Modello salvato in transformer_model.txt
```

### Predict

Dopo aver eseguito il training, per usare il modello senza riaddestrarlo:

```powershell
.\transformer_encoder.exe predict
```

La modalità `predict` carica `transformer_model.txt`, esegue soltanto il forward pass e stampa le predizioni:

```text
Target: 0, predizione: 0
Target: 1, predizione: 1
Accuratezza: 4/4
```

I valori esatti della loss possono cambiare se si modificano le inizializzazioni o il dataset.

## Cosa succede durante una run

Le modalità sono indipendenti e usano file diversi:

1. `train` legge `train_data.txt`, inizializza i pesi, addestra encoder e classificatore per `1000` epoche e salva il modello;
2. `predict` legge `transformer_model.txt` e `test_data.txt`, crea le rappresentazioni con il forward pass e classifica gli esempi.

Il comando:

```powershell
.\transformer_encoder.exe train
```

non esegue predizioni alla fine del training. Per classificarli bisogna eseguire separatamente:

```powershell
.\transformer_encoder.exe predict
```

Ogni nuova esecuzione di `train` riparte da pesi iniziali nuovi e sovrascrive `transformer_model.txt`. Una nuova esecuzione di `predict`, invece, riutilizza il modello già salvato e non modifica i pesi.

Se si esegue `predict` prima di `train`, il programma mostra un errore perché `transformer_model.txt` non esiste ancora. Il dataset di test è separato da quello di training, quindi l'accuratezza misura il comportamento su esempi non usati per aggiornare i pesi.

I comandi disponibili sono:

```text
programma train
programma predict
```

Il file dei pesi è testuale e contiene matrici, vettori `gamma`/`beta`, bias e parametri del classificatore.

## Formato dei dataset

I file `train_data.txt` e `test_data.txt` usano lo stesso formato testuale. La prima riga contiene il numero di esempi. Ogni esempio è composto da:

```text
label numero_token numero_feature
token_1_feature_1 token_1_feature_2 ...
token_2_feature_1 token_2_feature_2 ...
...
```

Per esempio:

```text
1
0 3 4
1.0 0.0 1.0 0.0
0.9 0.1 0.9 0.1
1.0 0.0 0.8 0.2
```

Il programma attuale richiede quattro feature per token. Il numero di token può variare, perché il mean pooling riassume sequenze di lunghezza diversa.

Per aggiungere esempi di training, modificare `train_data.txt`. Per aggiungere esempi di valutazione, modificare `test_data.txt`. I due file devono usare le stesse dimensioni delle feature e le label devono essere `0` o `1`.

## Cosa fa il programma

Il programma classifica sequenze molto piccole appartenenti a due classi:

- classe `0`: vettori prevalentemente del tipo `[1, 0, 1, 0]`;
- classe `1`: vettori prevalentemente del tipo `[0, 1, 0, 1]`.

Ogni esempio del dataset contiene attualmente tre token e ogni token ha quattro feature. I token sono già rappresentati come vettori numerici: non c'è un tokenizer e non c'è un vocabolario testuale.

Il flusso del modello è:

```text
vettori dei token
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

Per ogni esempio il programma esegue:

1. calcolo del forward pass;
2. mean pooling dell'output dell'encoder;
3. calcolo dei logits del classificatore;
4. softmax e cross-entropy;
5. calcolo dei gradienti nel backward pass;
6. aggiornamento dei pesi con gradient descent.

La struttura `EncoderCache` conserva i valori intermedi del forward, come `Q`, `K`, `V`, le probabilità dell'attenzione e le attivazioni della rete feed-forward. Questi valori servono per calcolare i gradienti senza ripetere il forward.

I parametri principali del training sono definiti in `main`:

```cpp
const double learningRate = 0.01;
const std::size_t epochs = 1000;
```

Per un training più lento o più veloce si può modificare `learningRate`. Per eseguire più passaggi sul dataset si può aumentare `epochs`.

## File del progetto

- `main.cpp`: implementazione dell'encoder, del backward pass, del classificatore e del training;
- `theory.md`: spiegazione teorica di Xavier initialization, loss, cross-entropy, pooling, cache, layer normalization, `gamma` e `beta`;
- `train_data.txt`: esempi usati dalla modalità `train`;
- `test_data.txt`: esempi usati dalla modalità `predict`;
- `transformer_model.txt`: pesi salvati da `train` e caricati da `predict`;
- `transformer_encoder.exe`: eseguibile generato dalla compilazione, se si usa il comando mostrato sopra.

## Limiti dell'esempio

Questo programma è intenzionalmente minimale:

- usa una sola testa di attenzione;
- usa pesi inizializzati casualmente ma con generatori deterministici;
- non legge testo: i dataset contengono già vettori numerici;
- non separa dataset di training e validation;
- non implementa dropout, maschere di attenzione o batch training;
- usa matrici costruite con `std::vector`, quindi non è ottimizzato per sequenze grandi;
- il training è pensato per essere eseguito su PC, non direttamente su Arduino.

Per usare il modello su Arduino, un possibile passo successivo sarebbe addestrare i pesi sul PC, salvarli in un formato compatibile e implementare sull'Arduino soltanto il forward pass.
