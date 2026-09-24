# Teoria del Transformer e del training

Questo documento parte dai concetti fondamentali e arriva all'implementazione
di `transformer_playground/main.cpp`. L'esempio usa un Transformer encoder
molto piccolo, scritto in C++ senza librerie esterne, che classifica parole
ASCII come esistenti o non esistenti.

## 1. AI, machine learning, deep learning e modelli linguistici

**Intelligenza artificiale (AI)** è un nome generale per tecniche che fanno
svolgere a un computer compiti che associamo all'intelligenza: riconoscere
immagini, tradurre, classificare testi, trovare anomalie o prendere decisioni.

**Machine learning** è un modo di costruire questi sistemi: invece di scrivere
tutte le regole a mano, forniamo esempi e lasciamo che il programma impari
alcune regole dai dati.

**Deep learning** è machine learning basato su reti neurali con molti strati.
Una rete neurale trasforma numeri in altri numeri usando parametri, cioè valori
modificati durante l'apprendimento.

Un **modello linguistico** lavora con il linguaggio e spesso cerca di prevedere
il token successivo:

```text
Il gatto beve il ___ -> latte
```

Un **LLM** (*Large Language Model*) è un modello linguistico grande, addestrato
su molti testi. Molti LLM usano Transformer, ma non tutti i sistemi di AI sono
Transformer e non tutti i Transformer sono LLM.

## 2. Da una sequenza a una matrice di numeri

Un Transformer non lavora direttamente con parole o concetti: riceve una
matrice numerica. Se la sequenza contiene `N` elementi e ogni elemento è un
vettore di dimensione `d_model`, la forma è:

```text
N righe × d_model colonne
```

Nel caso testuale, ogni riga può rappresentare un token. Un token può essere
una parola intera, una parte di parola, un segno di punteggiatura o un simbolo
speciale. Non è necessariamente una parola completa.

Nel progetto, i token sono i caratteri ASCII della parola. Ogni carattere viene
convertito in una riga della tabella di character embedding:

```text
'c' -> [0.2, -0.1, 0.7, ...]  // 10 dimensioni
'a' -> [0.4,  0.3, 0.1, ...]
```

La tabella ha forma `128 × 10`: una riga per ogni carattere ASCII e dieci
coordinate apprendibili per carattere. Non c'è un tokenizer linguistico o un
vocabolario di parole; ogni carattere ASCII è direttamente un token.

### Vettori e dimensioni

Una riga è un vettore. Le colonne sono coordinate numeriche comuni a tutti gli
elementi:

```text
          dim. 0   dim. 1   dim. 2
token 0     0.2      0.7     -0.1
token 1    -0.4      0.3      0.8
```

Nel caso di dati grezzi, una colonna può avere un significato esplicito, come
temperatura o velocità. In un embedding o in una rappresentazione interna del
Transformer, invece, le dimensioni sono coordinate latenti: normalmente non
hanno un significato umano isolato. È la combinazione delle dimensioni a
rappresentare pattern utili.

## 3. Embedding e posizione

Nel testo un token id è soltanto un indice, per esempio:

```text
"gatto" -> 42
```

Il modello usa l'id per recuperare una riga da una tabella di embedding:

```text
embedding[42] -> [0.2, -0.7, 0.4, 1.1]
```

L'embedding è la rappresentazione iniziale del token. Non indica la sua
posizione nella sequenza.

Nel codice, gli embedding hanno dimensione 10 e sono inizializzati casualmente
con Xavier. Durante il training vengono aggiornate anche le righe relative ai
caratteri usati nella parola.

### Positional encoding

La self-attention deve ricevere anche l'ordine dei token. Senza informazione
posizionale, sequenze come queste conterrebbero gli stessi elementi ma in un
ordine indistinguibile:

```text
il cane morde l'uomo
l'uomo morde il cane
```

Il modo originale usa un **positional encoding sinusoidale**, sommato
all'embedding:

```text
input = embedding + encoding_della_posizione
```

Esistono anche altre tecniche:

- **embedding posizionale appreso**: un vettore addestrabile per ogni posizione;
- **posizione relativa**: rappresenta la distanza tra due token;
- **RoPE**: ruota `Q` e `K` in base alla posizione;
- **ALiBi**: aggiunge un bias ai punteggi dell'attenzione in base alla distanza.

Il progetto usa una versione sinusoidale, calcolata dalla funzione
`positionalEncoding`.

## 4. Self-attention: l'idea centrale

La self-attention permette a ogni riga di guardare le altre righe e decidere
quali siano importanti. L'elemento alla posizione `i` confronta la propria
richiesta con tutti gli altri elementi e costruisce una nuova rappresentazione
che contiene informazioni dal contesto.

Per ogni elemento si calcolano tre vettori:

- **Query (`Q`)**: che tipo di informazione sto cercando?
- **Key (`K`)**: che tipo di informazione offro per essere trovato?
- **Value (`V`)**: quale contenuto trasferisco se vengo selezionato?

Non sono tre proprietà semantiche scritte a mano: sono proiezioni numeriche
della rappresentazione corrente.

### Come si ottengono Q, K e V

Se `X` è la matrice di input:

```text
Q = X · WQ
K = X · WK
V = X · WV
```

`WQ`, `WK` e `WV` sono matrici di pesi diverse. Ogni riga `x_i` diventa quindi
una query `q_i`, una key `k_i` e un value `v_i`:

```text
x_i -> q_i, k_i, v_i
```

Il value è il contenuto offerto dal token nella sua rappresentazione corrente.
Non è una proprietà fissa della parola: nei blocchi successivi la
rappresentazione corrente contiene già informazioni contestuali.

### Punteggi e softmax

Per il token `i`, il punteggio rispetto al token `j` è:

```text
score(i, j) = Q_i · K_j / sqrt(d_k)
```

La divisione per `sqrt(d_k)` mantiene i punteggi in una scala gestibile. La
**softmax** trasforma i punteggi in pesi che sono compresi tra 0 e 1 e sommano
a 1:

```text
softmax(x_i) = exp(x_i) / somma_j exp(x_j)
```

Esempio:

```text
punteggi     = [2, 1, 0]
softmax      = [0.665, 0.245, 0.090]
```

Il token con il punteggio più alto riceve più peso, ma anche gli altri possono
contribuire. Infine i pesi combinano i value:

```text
output_i = alpha_i0 * V_0 + alpha_i1 * V_1 + ...
```

La formula compatta è:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Nel codice la sottrazione del massimo prima dell'esponenziale serve soltanto
a evitare problemi numerici e non cambia la softmax finale.

## 5. Una head e più head

Una **attention head** è un'attenzione completa, con la propria proiezione di
`Q`, `K` e `V`:

```text
Q_1, K_1, V_1 -> head_1
Q_2, K_2, V_2 -> head_2
```

La **multi-head attention** esegue più attenzioni in parallelo. Le head possono
imparare relazioni diverse, anche se non ricevono istruzioni esplicite del tipo
"questa head cerca i verbi". Le uscite vengono concatenate e proiettate:

```text
output = Concat(head_1, head_2, ...) · WO
```

Il progetto usa una sola head per mantenere l'implementazione leggibile.

### GQA e MQA

In **Grouped-Query Attention (GQA)** più query head condividono le stesse key
e value:

```text
Q_1 ─┐
Q_2 ─┘ -> K_1, V_1

Q_3 ─┐
Q_4 ─┘ -> K_2, V_2
```

Le query fanno domande diverse allo stesso archivio di rappresentazioni. Anche
con gli stessi `K/V`, producono pesi diversi e quindi combinazioni diverse dei
value. In **Multi-Query Attention (MQA)** tutte le query condividono una sola
coppia `K/V`. La condivisione riduce memoria e costo durante la generazione.

## 6. Residual connection

Una residual connection è una scorciatoia che somma l'input al risultato di un
sottoblocco:

```text
output = input + trasformazione(input)
```

Nel Transformer può avvolgere self-attention, feed-forward e, nei modelli
encoder-decoder, cross-attention:

```text
x1 = x  + SelfAttention(x)
x2 = x1 + FeedForward(x1)
x3 = x2 + CrossAttention(x2, encoder_output)
```

La trasformazione aggiunge un aggiornamento senza dover ricreare da zero la
rappresentazione. Le due matrici devono avere la stessa forma; altrimenti si
usa una proiezione per renderle compatibili.

Nel progetto:

```cpp
cache.attentionResidual = add(input, cache.context);
cache.outputResidual = add(cache.normalizedAttention, cache.feedForwardOutput);
```

## 7. Layer normalization

LayerNorm normalizza le dimensioni di ogni riga, quindi ogni token o elemento
della sequenza separatamente. Non modifica la tabella degli embedding: agisce
sulla rappresentazione temporanea dopo le trasformazioni.

Per un vettore `x`:

```text
normalized = (x - media) / sqrt(varianza + epsilon)
```

È utile perché attenzione, residual connection e feed-forward possono cambiare
la scala dei valori. Nei modelli reali si aggiungono parametri apprendibili:

```text
output = gamma * normalized + beta
```

Il progetto implementa sia la normalizzazione sia `gamma` e `beta`, che vengono
aggiornati durante il training.

## 8. Feed-forward network

Dopo aver raccolto informazioni dagli altri elementi, ogni riga viene elaborata
indipendentemente dalle altre con una piccola rete neurale:

```text
FFN(x) = Linear_2(ReLU(Linear_1(x)))
```

La prima trasformazione espande la dimensione, per esempio:

```text
4 -> 8 -> 4
```

Le otto componenti intermedie sono combinazioni apprese delle quattro
componenti originali. La ReLU applica:

```text
ReLU(z) = max(0, z)
```

Spegne i valori negativi e lascia invariati quelli positivi. Non decide da
sola cosa sia irrilevante: i pesi appresi fanno emergere combinazioni utili.
La seconda trasformazione ricompone le caratteristiche e torna a `d_model`,
così il risultato può essere sommato tramite la residual connection.

La self-attention mescola informazioni tra elementi; la FFN elabora
localmente ogni elemento usando le informazioni raccolte.

## 9. Il blocco Transformer e la profondità

Un blocco encoder completo è quindi:

```text
input
  ↓
self-attention
  ↓
residual connection + LayerNorm
  ↓
feed-forward: Linear -> ReLU -> Linear
  ↓
residual connection + LayerNorm
  ↓
output del blocco
```

I modelli reali impilano molti blocchi simili. La struttura generale è la
stessa, ma ogni blocco ha pesi diversi. Il numero di blocchi è la profondità o
il numero di layer. Il progetto ne usa uno solo.

## 10. Encoder, decoder, BERT e T5

Il Transformer originale aveva encoder e decoder. Oggi il termine Transformer
indica una famiglia di varianti:

```text
encoder-only   -> comprensione e rappresentazione
decoder-only   -> generazione autoregressiva
encoder-decoder -> trasformazione testo -> testo
```

Un encoder può guardare a sinistra e a destra nella sequenza. Un decoder usa
una causal mask e non può guardare i token futuri.

**BERT** è encoder-only. Nel pre-training nasconde token e prova a ricostruirli
usando il contesto da entrambe le direzioni:

```text
Il gatto [MASK] il latte -> beve
```

Per classificare il sentiment si aggiunge una classification head, spesso
basata sulla rappresentazione del token speciale `[CLS]`.

**T5** (*Text-to-Text Transfer Transformer*) usa encoder e decoder. Trasforma
ogni compito in testo in ingresso e testo in uscita:

```text
translate English to German: Hello -> Hallo
classify sentiment: This film is fantastic -> positive
```

Durante il pre-training T5 nasconde spesso intervalli interi di testo e il
decoder genera il testo mancante, usando token speciali come `<extra_id_0>`.

## 11. Output, logits e classificazione

L'encoder produce una rappresentazione per ogni riga. Per classificare
l'intera sequenza, il progetto fa **mean pooling**: calcola la media delle
rappresentazioni e ottiene un solo vettore.

Poi una output head trasforma quel vettore nello spazio delle risposte:

```text
vettore finale -> logits per ogni classe
```

I **logits** sono punteggi grezzi. Non sono probabilità e non devono sommare a
1. Una softmax può trasformarli in probabilità:

```text
logits       = [2.0, 1.0, 0.0]
softmax      = [0.665, 0.245, 0.090]
```

Se serve soltanto scegliere la classe più probabile, si può usare direttamente
`argmax(logits)`: la softmax conserva l'ordine dei valori. Nel progetto la
softmax serve per la cross-entropy durante il training e per interpretare le
probabilità.

La parola **vocabolario** vale solo per i modelli linguistici: in quel caso
l'output head produce un logit per ogni token possibile. In un classificatore
come questo produce invece un logit per ogni classe.

## 12. Forward pass, training e inference

Il **forward pass** calcola un output partendo dall'input:

```text
input -> encoder -> pooling -> logits
```

Durante il **training** si confrontano i logits con la classe corretta usando
una loss. La backpropagation calcola i gradienti e un ottimizzatore aggiorna i
pesi.

Durante l'**inference** si usano i pesi già appresi senza aggiornarli:

```text
architettura + pesi addestrati -> modello utile
architettura + pesi casuali     -> sola dimostrazione del calcolo
```

Il progetto implementa entrambi i percorsi: `train` aggiorna i pesi e salva il
modello; `predict` li carica e classifica il dataset di test.

## 13. Limiti dell'esempio

Questo è un laboratorio didattico, non un LLM. In particolare:

- usa una sola attention head;
- usa un solo blocco encoder;
- legge parole ASCII ma non usa un tokenizer linguistico;
- usa una tabella di embedding `128 × 10`;
- classifica un piccolo dataset di parole, non verifica un dizionario reale;
- non usa dropout, batching o maschere causali;
- implementa il training manualmente e senza ottimizzatori avanzati;
- è scritto con `std::vector`, quindi non è ottimizzato per modelli grandi.

Il suo scopo è rendere visibile il percorso completo:

```text
vettori numerici
  -> positional encoding
  -> self-attention
  -> residual + LayerNorm
  -> feed-forward
  -> residual + LayerNorm
  -> mean pooling
  -> classificatore
  -> loss
  -> backpropagation
```

## 14. Mappa rapida tra concetti e codice

| Concetto | Dove compare |
|---|---|
| Character embedding | `CharacterEmbedding`, `embedding.forward` |
| Positional encoding | `positionalEncoding` |
| Query, key, value | `cache.query`, `cache.key`, `cache.value` |
| Self-attention | `cache.attentionScores`, `softmaxRows`, `cache.context` |
| Residual connection | `attentionResidual`, `outputResidual` |
| LayerNorm | `layerNorm`, `applyLayerNormAffine` |
| Feed-forward | `W1`, `W2`, `relu` |
| Mean pooling | `meanRows` |
| Logits | `LinearClassifier::predictLogits` |
| Loss | `-std::log(probabilities[target])` |
| Backpropagation | `Encoder::backward` e `LinearClassifier::train` |
| Salvataggio | `saveModel`, `Encoder::save`, `LinearClassifier::save` |

## 15. Inizializzazione Xavier

La **Xavier initialization**, chiamata anche **Glorot initialization**, è un metodo per inizializzare i pesi di una rete neurale prima dell'addestramento.

### Perché serve

I pesi iniziali non dovrebbero essere tutti uguali, altrimenti i neuroni imparerebbero esattamente la stessa cosa. Tuttavia, scegliere valori casuali troppo grandi o troppo piccoli può creare problemi:

- valori troppo grandi possono far esplodere le attivazioni e i gradienti;
- valori troppo piccoli possono farli diventare quasi nulli;
- in entrambi i casi il training può diventare instabile o molto lento.

Xavier sceglie la scala dei pesi in base a due quantità:

- `fan_in`: numero di ingressi del neurone;
- `fan_out`: numero di uscite del neurone.

In questo modo cerca di mantenere simile la varianza dei dati mentre attraversano i vari strati della rete.

### Distribuzione uniforme

Con una distribuzione uniforme, ogni peso viene estratto dall'intervallo:

$$
W_{ij} \sim U\left(-\sqrt{\frac{6}{fan\_in + fan\_out}},
+\sqrt{\frac{6}{fan\_in + fan\_out}}\right)
$$

Dove `U(a, b)` indica una distribuzione uniforme tra `a` e `b`.

Per esempio, se `fan_in = 4` e `fan_out = 8`:

$$
limit = \sqrt{\frac{6}{4 + 8}} = \sqrt{0.5} \approx 0.707
$$

Ogni peso viene quindi inizializzato casualmente tra `-0.707` e `+0.707`.

### Distribuzione normale

Una variante usa una distribuzione normale con media zero e deviazione standard:

$$
\sigma = \sqrt{\frac{2}{fan\_in + fan\_out}}
$$

Quindi:

$$
W_{ij} \sim N(0, \sigma^2)
$$

Dove `N(0, sigma^2)` indica una distribuzione normale con media zero e varianza `sigma^2`.

### Esempio in C++

```cpp
#include <cmath>
#include <random>

Matrix xavierWeights(std::size_t fanIn,
                     std::size_t fanOut) {
    const double limit = std::sqrt(
        6.0 / static_cast<double>(fanIn + fanOut));

    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_real_distribution<double> distribution(-limit, limit);

    Matrix weights(fanIn, std::vector<double>(fanOut));
    for (auto& row : weights) {
        for (double& weight : row) {
            weight = distribution(generator);
        }
    }
    return weights;
}
```

La funzione restituisce una matrice con `fanIn` righe e `fanOut` colonne. Nel Transformer, per esempio, può essere usata per inizializzare le matrici `Wq`, `Wk`, `Wv`, `W1` e `W2`.

### Nel Transformer

Per una proiezione lineare come:

```cpp
Q = input * Wq;
```

se `input` ha dimensione `dModel` e `Q` ha ancora dimensione `dModel`, allora:

```text
fan_in  = dModel
fan_out = dModel
```

Per il primo strato della rete feed-forward, se la dimensione passa da `dModel` a `feedForwardSize`:

```text
fan_in  = dModel
fan_out = feedForwardSize
```

Xavier determina soltanto i valori iniziali. Non sostituisce il training: durante la backpropagation i pesi vengono aggiornati usando i gradienti.

### Limite pratico

Xavier è particolarmente adatta alle funzioni di attivazione simmetriche, come `tanh`. Con `ReLU` si usa spesso la **He initialization**, che tiene conto del fatto che ReLU annulla i valori negativi.

### Perché non usare direttamente scalari per i caratteri

Se vogliamo classificare una stringa, possiamo assegnare un ID intero a ogni carattere:

```text
a -> 1
b -> 2
c -> 3
...
```

Questi ID sono utili come indici, ma non dovrebbero essere passati direttamente al Transformer come valori numerici. Lo scalare `20`, infatti, non contiene soltanto l'identità del carattere: introduce anche un ordine e una distanza tra i caratteri.

Con una rappresentazione scalare, il modello può interpretare erroneamente che:

```text
distanza(a, b) < distanza(a, z)
```

perché, usando gli ID precedenti, `|1 - 2| < |1 - 26|`. Ma dal punto di vista linguistico non è detto che `a` sia più simile a `b` che a `z`. L'ordine numerico dipende solo dalla numerazione scelta: se scambiassimo gli ID, cambierebbero anche le distanze senza cambiare il significato dei caratteri.

Il problema diventa evidente nelle proiezioni lineari del Transformer. Se `x` è uno scalare, una proiezione come:

$$
q = xW_q
$$

produce un vettore la cui ampiezza dipende direttamente dall'ID assegnato. Anche i punteggi dell'attenzione:

$$
score(q,k) = \frac{q \cdot k}{\sqrt{d_k}}
$$

dipenderebbero dai valori arbitrari degli ID. Il modello dovrebbe prima imparare a ignorare questo ordine artificiale, rendendo il training più difficile e la rappresentazione meno stabile.

#### ID come indice, embedding come input

Usare gli ID non è sbagliato in assoluto. La distinzione importante è questa:

```text
ID intero -> indice in una tabella -> vettore embedding -> Transformer
```

La tabella di embedding contiene un vettore appreso per ogni carattere:

```text
'a' -> [ 0.12, -0.45,  0.77, ...]
'b' -> [-0.31,  0.20,  0.11, ...]
'z' -> [ 0.08,  0.91, -0.36, ...]
```

In questo caso gli ID servono soltanto per recuperare le righe della tabella. Il Transformer riceve i vettori, non i numeri `1`, `2` o `26`. Durante il training gli embedding vengono aggiornati insieme agli altri pesi, quindi il modello può imparare quali caratteri siano utili da considerare simili per il compito.

Un'alternativa è il **one-hot encoding**, in cui ogni carattere è rappresentato da un vettore con un solo `1` e tutti gli altri valori uguali a zero. Il one-hot non introduce un ordine artificiale, ma ha dimensione pari al numero di caratteri e non contiene similarità apprese. Una matrice di embedding applicata a un vettore one-hot equivale, in pratica, a selezionare la riga corrispondente della tabella.

Per una sequenza di caratteri il flusso corretto è quindi:

```text
caratteri -> ID interi -> character embedding -> positional encoding -> Transformer
```

#### Come scegliere la dimensione degli embedding

Non esiste una formula universale che stabilisca la dimensione ottimale di un embedding. La dimensione è un **iperparametro**: deve essere scelta in base al problema e verificata sperimentalmente.

La dimensione dell'embedding non corrisponde al numero minimo di bit necessario per rappresentare l'ID. Se il vocabolario contiene 30 caratteri, bastano 5 bit per distinguere gli ID:

$$
\lceil \log_2(30) \rceil = 5
$$

Ma un embedding non è un codice binario. È un vettore di numeri reali appresi, usato per rappresentare proprietà utili al compito. Per questo 5 bit e 5 dimensioni di embedding sono concetti diversi, anche se 5 dimensioni possono comunque essere una scelta ragionevole per un problema piccolo.

La procedura più affidabile è confrontare più dimensioni candidate mantenendo uguali dataset, architettura, learning rate ed epoche. Per esempio:

```text
dimensioni candidate: 4, 5, 8, 16
```

Per ogni candidata si addestra un modello e si misurano almeno:

- loss e accuratezza sul training set;
- loss e accuratezza su un validation set separato;
- stabilità del risultato con più inizializzazioni casuali;
- numero di parametri e tempo di addestramento.

La validation è importante perché una dimensione grande può permettere al modello di memorizzare il training set senza imparare regole generalizzabili. I casi tipici sono:

```text
training loss alta, validation loss alta
-> rappresentazione probabilmente troppo piccola o modello insufficiente

training loss bassa, validation loss alta
-> possibile overfitting, rappresentazione o modello troppo complessi

training loss bassa, validation loss bassa
-> buona capacità di apprendere e generalizzare
```

Per questo esempio il vocabolario contiene 128 caratteri ASCII e la dimensione
dell'embedding è fissata a `10`. Se l'embedding è anche l'input del Transformer,
si imposta:

```text
embedding dimension = modelSize
```

così non serve una proiezione aggiuntiva. Per esempio, una configurazione didattica potrebbe essere:

```text
vocabolario: 128 caratteri ASCII
embedding dimension: 10
modelSize: 10
feedForwardSize: 10
```

La scelta finale dovrebbe essere la dimensione più piccola che raggiunge buone prestazioni sul validation set in modo stabile. In questo modo si limita il numero di parametri senza sacrificare la capacità del modello.

## 16. Concetti fondamentali del training

### Loss

La **loss** è una misura numerica di quanto le predizioni del modello siano lontane dai valori corretti, chiamati target o label.

Durante il training il modello cerca di minimizzare la loss. Una loss vicina a zero indica che, sugli esempi considerati, le predizioni sono generalmente corrette; non garantisce però che il modello funzioni bene su dati nuovi.

Nel programma, la loss di ogni esempio viene sommata e poi divisa per il numero di esempi:

$$
loss_{media} = \frac{1}{N}\sum_{i=1}^{N} loss_i
$$

### Cross-entropy

La **cross-entropy** è una loss usata spesso nei problemi di classificazione. Il modello produce un punteggio per ogni classe, chiamato logit. La softmax trasforma questi punteggi in probabilità:

$$
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

Dove `z_j` è il logit della classe `j` e `p_j` è la probabilità assegnata a quella classe.

Se la classe corretta è `y`, la cross-entropy per un esempio è:

$$
loss = -\log(p_y)
$$

La penalità è piccola quando il modello assegna alta probabilità alla classe corretta. Se invece assegna probabilità quasi zero alla classe corretta, la loss diventa grande.

Nel codice questa operazione appare come:

```cpp
double loss = -std::log(probabilities[target]);
```

La combinazione softmax + cross-entropy produce anche un gradiente semplice rispetto ai logits:

$$
\frac{\partial loss}{\partial z_j} = p_j - 1[j = y]
$$

Dove `1[j = y]` vale `1` quando `j` è la classe corretta e `0` negli altri casi.

### Discesa del gradiente

La **discesa del gradiente** aggiorna i pesi nella direzione che riduce la loss. Se `w` è un peso e `g` è il gradiente della loss rispetto a quel peso, l'aggiornamento è:

$$
w_{nuovo} = w_{vecchio} - \eta g
$$

`eta` è il learning rate, cioè la velocità di apprendimento. Un valore troppo grande può rendere il training instabile; un valore troppo piccolo può renderlo molto lento.

Nel classificatore del programma, per ogni classe `j` il gradiente del peso associato alla feature `x_i` è:

$$
\frac{\partial loss}{\partial W_{ij}} = x_i(p_j - 1[j = y])
$$

Il classificatore aggiorna quindi i propri pesi dopo ogni esempio. Un ciclo completo su tutti gli esempi del dataset si chiama epoca.

### Ottimizzatore con momentum

Il programma usa una forma semplice di **stochastic gradient descent** (SGD):
dopo ogni esempio aggiorna direttamente ogni peso usando il gradiente corrente:

$$
w_t = w_{t-1} - \eta g_t
$$

Dove `g_t` è il gradiente al passo `t` e `eta` è il learning rate. Questo
aggiornamento può essere rumoroso, perché il gradiente di una singola parola
può puntare in una direzione diversa da quello dell'esempio successivo.

Il **momentum** aggiunge una memoria degli aggiornamenti precedenti. Una forma
comune delle equazioni è:

$$
v_t = \beta v_{t-1} + g_t
$$

$$
w_t = w_{t-1} - \eta v_t
$$

`v_t` è la velocità o direzione accumulata, mentre `beta` controlla quanta
memoria conservare. Con `beta = 0.9`, per esempio, gli aggiornamenti recenti
continuano a influenzare il movimento del peso.

L'effetto intuitivo è simile a una pallina che scende lungo una superficie:

- se molti gradienti consecutivi puntano nella stessa direzione, il momentum
  accelera il movimento;
- se i gradienti oscillano avanti e indietro, la memoria li smorza;
- il training può diventare più veloce e meno rumoroso.

Il momentum richiede però di conservare un vettore `v` della stessa forma di
ogni matrice o vettore di pesi. Per questo, quando si salva un checkpoint per
continuare il training, un ottimizzatore con momentum dovrebbe salvare sia i
pesi sia i relativi buffer `v`. Il progetto attuale non usa momentum, quindi il
resume salva e ricarica soltanto i pesi del modello.

### Checkpoint, ultimo modello e modello migliore

Durante un training lungo è utile salvare il modello in punti intermedi. Un
**checkpoint** è una fotografia dei pesi e dello stato necessario per
continuare il lavoro.

Il progetto mantiene due checkpoint distinti:

- `transformer_model_latest.txt`: contiene i pesi dell'ultima epoca completata
  ed è quello caricato da `--resume`;
- `transformer_model.txt`: contiene i pesi dell'epoca con la validation loss più
  bassa ed è quello usato da `predict`.

Entrambi salvano anche alcuni metadati:

- numero dell'ultima epoca completata;
- numero dell'epoca migliore;
- training loss;
- validation loss, quando esiste.

Questo permette a una nuova run di riprendere il conteggio globale delle
epoche. Per esempio, dopo due epoche una run con `--resume --epochs 1` stampa
`Inizio Epoca #3` invece di ricominciare da `#1`.

Con un ottimizzatore dotato di momentum, nei checkpoint andrebbero salvati
anche i buffer di momentum. Nel progetto attuale non sono necessari perché
viene usata la SGD semplice.

### Come scegliere il numero di epoche

Non esiste un numero di epoche corretto in assoluto. Dipende dalla dimensione
del dataset, dalla difficoltà del compito, dal learning rate e dalla capacità
del modello. Un'epoca in più non significa automaticamente un modello migliore:
oltre un certo punto il modello può iniziare a **memorizzare** gli esempi di
training invece di imparare regole utili per esempi nuovi.

Il metodo standard usa tre suddivisioni dei dati:

- **training set**: viene usato per aggiornare i pesi;
- **validation set**: non aggiorna i pesi e serve per scegliere iperparametri,
  come il numero di epoche;
- **test set**: viene usato soltanto alla fine per una valutazione finale.

Dopo ogni epoca si misurano almeno la loss sul training set e quella sul
validation set. I casi tipici sono:

```text
training loss alta, validation loss alta
    -> il modello non ha ancora imparato abbastanza: underfitting

training loss diminuisce, validation loss diminuisce
    -> il training sta migliorando anche sui dati non visti

training loss diminuisce, validation loss aumenta
    -> il modello sta iniziando a fare overfitting
```

Una strategia comune è l'**early stopping**: si salva il modello ogni volta
che la validation loss migliora e si interrompe il training quando non migliora
più per un certo numero di epoche consecutive, chiamato *patience*. Per
esempio, con `patience = 5` si può interrompere dopo cinque epoche senza
miglioramenti, mantenendo i pesi dell'epoca migliore.

Nel progetto la validation è opzionale. Si può passare uno o più file con
`--validation`; questi esempi vengono valutati dopo ogni epoca senza aggiornare
i pesi. Per esempio:

```text
programma train dataset1 dataset2 --validation validation.txt --epochs N --patience 5
```

L'opzione `--validation` può essere ripetuta per concatenare più file. Il
programma salva in memoria i pesi dell'epoca con la validation loss più bassa.
Se la validation loss non migliora per `patience` epoche consecutive, il
training termina e vengono ripristinati i pesi migliori.

`test_data.txt` viene usato da `predict` dopo il training e va considerato un
vero test set, a meno che non venga passato esplicitamente a `--validation`.
Non conviene usare ripetutamente il test set per decidere quante epoche
scegliere. Il progetto offre inoltre un criterio più semplice: si può
interrompere il training quando la loss media sul training set scende sotto una
soglia configurabile:

```text
programma train dataset1 dataset2 --epochs N --loss-threshold X
```

Per esempio, con `--loss-threshold 0.01` il programma completa l'epoca
corrente e non ne avvia altre se la sua loss media è inferiore a `0.01`. Questa
regola può evitare epoche inutili, ma guarda soltanto il training set: una loss
molto bassa può essere anche un segnale di overfitting. Quando disponibile,
l'early stopping basato sulla validation loss è quindi più affidabile.

Il numero massimo di epoche resta configurabile con `--epochs N` e funziona
come limite superiore: il training si ferma quando raggiunge `N` epoche oppure,
prima, quando viene rispettata la soglia della loss.

Con il dataset didattico piccolo si possono usare molte epoche. Con le oltre
700.000 parole Morph-it! positive e negative è più ragionevole iniziare da
`1` o poche epoche, osservare la loss e poi aumentare gradualmente se il
modello non ha ancora imparato. Anche una sola epoca sul dataset completo può
richiedere molto tempo con questa implementazione, perché gli esempi vengono
processati uno alla volta e senza batch.

### Mean pooling

L'encoder produce una rappresentazione per ogni token. Se una sequenza contiene `N` token e ogni rappresentazione ha dimensione `d`, l'output è una matrice `N x d`:

$$
H =
\begin{bmatrix}
h_1 \\
h_2 \\
\vdots \\
h_N
\end{bmatrix}
$$

Per classificare l'intera sequenza serve una singola rappresentazione. Il **mean pooling**, o pooling medio, calcola la media delle rappresentazioni dei token:

$$
h_{medio} = \frac{1}{N}\sum_{i=1}^{N} h_i
$$

Il vettore `h_medio` ha ancora dimensione `d`, ma riassume tutta la sequenza. Nel programma viene calcolato dalla funzione `meanRows` e passato al classificatore lineare.

### Cache del forward pass

Durante il **forward pass** il modello calcola l'output partendo dall'input. Per esempio, nell'encoder calcola `Q`, `K`, `V`, i punteggi dell'attenzione, le probabilità della softmax, l'output dell'attenzione e i valori intermedi della rete feed-forward.

Per calcolare i gradienti durante il **backward pass**, servono molti di questi valori intermedi. Per questo il forward li salva in una struttura chiamata `EncoderCache`:

```cpp
struct EncoderCache {
    Matrix input;
    Matrix query;
    Matrix key;
    Matrix value;
    Matrix attentionScores;
    Matrix attentionProbabilities;
    Matrix context;
    Matrix attentionResidual;
    Matrix attentionNormalized;
    Matrix normalizedAttention;
    Matrix feedForwardInput;
    Matrix feedForward;
    Matrix feedForwardOutput;
    Matrix outputResidual;
    Matrix outputNormalized;
    Matrix output;
};
```

La cache non contiene nuovi pesi e non viene addestrata. È soltanto una fotografia dei calcoli fatti per un singolo esempio. Il backward la usa per applicare la regola della catena:

```text
output della loss
    -> output encoder
    -> feed-forward
    -> layer normalization
    -> attenzione
    -> Q, K, V
    -> pesi
```

Salvare questi valori evita di ricalcolare il forward durante il backward e rende possibile calcolare, per esempio, i gradienti di `Wq` a partire da `input` e dal gradiente di `Q`:

$$
\frac{\partial loss}{\partial W_q} = input^T \frac{\partial loss}{\partial Q}
$$

La cache deve vivere almeno fino alla fine del backward relativo all'esempio. Dopo l'aggiornamento dei pesi può essere eliminata e sostituita dalla cache del successivo forward pass.

### Gamma e beta della layer normalization

La layer normalization normalizza ogni riga, cioè ogni rappresentazione di token, usando la media e la varianza delle sue feature. Dato un vettore `x` di dimensione `d`:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
$$

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

Prima si ottiene il vettore normalizzato:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

La versione completa applica poi due parametri addestrabili, `gamma` e `beta`:

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

### Cosa significa affine

Una trasformazione **affine** combina una moltiplicazione e una somma:

$$
f(x) = ax + b
$$

Nel caso della layer normalization, la trasformazione viene applicata separatamente a ogni feature:

- `gamma` è il fattore di scala `a`;
- `beta` è lo spostamento `b`.

Quindi la normalizzazione produce prima `x_hat`, con media circa zero e varianza circa uno. La parte affine calcola poi `y`, modificando scala e posizione di ogni feature:

```text
x -> normalizzazione -> x_hat -> gamma * x_hat + beta -> y
```

La parte affine è importante perché la normalizzazione da sola impone una distribuzione standardizzata. Grazie a `gamma` e `beta`, il modello può imparare una scala diversa, uno spostamento diverso oppure lasciare quasi invariata una feature. Infatti, con `gamma = 1` e `beta = 0` si ottiene semplicemente `y = x_hat`.

In senso matematico, `gamma * x + beta` è affine e non soltanto lineare: se `beta` è diverso da zero, la trasformazione non conserva necessariamente lo zero. Nel Transformer `gamma` e `beta` sono vettori e ogni componente agisce sulla feature corrispondente; non sono un altro strato Transformer e non mescolano direttamente feature diverse.

`gamma` controlla la scala di ogni feature, mentre `beta` controlla il suo spostamento. Sono vettori della stessa dimensione del modello, non matrici:

```text
gamma = [gamma_1, gamma_2, ..., gamma_d]
beta  = [beta_1, beta_2, ..., beta_d]
```

Di solito vengono inizializzati così:

```text
gamma = [1, 1, ..., 1]
beta  = [0, 0, ..., 0]
```

Questa inizializzazione fa sì che all'inizio la layer normalization esegua soltanto la normalizzazione. Durante il training `gamma` e `beta` possono poi imparare quanta scala e quale spostamento siano utili per ogni feature. I loro gradienti sono:

$$
\frac{\partial loss}{\partial \gamma_i} = \sum_{token} \frac{\partial loss}{\partial y_i}\hat{x}_i
$$

$$
\frac{\partial loss}{\partial \beta_i} = \sum_{token} \frac{\partial loss}{\partial y_i}
$$

Nel nostro esempio la funzione `layerNorm` calcola la normalizzazione senza la parte affine. La classe `Encoder` applica poi `gamma` e `beta` in entrambe le layer normalization, li conserva tra un esempio e l'altro e li aggiorna durante il backward insieme agli altri pesi.

Il flusso completo è quindi:

```text
token -> encoder -> rappresentazione per token
            |
            v
           mean pooling
            |
            v
        classificatore lineare
            |
            v
            softmax
            |
            v
        cross-entropy (loss)
            |
            v
          discesa del gradiente
```
