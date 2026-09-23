# Xavier initialization

La **Xavier initialization**, chiamata anche **Glorot initialization**, è un metodo per inizializzare i pesi di una rete neurale prima dell'addestramento.

## Perché serve

I pesi iniziali non dovrebbero essere tutti uguali, altrimenti i neuroni imparerebbero esattamente la stessa cosa. Tuttavia, scegliere valori casuali troppo grandi o troppo piccoli può creare problemi:

- valori troppo grandi possono far esplodere le attivazioni e i gradienti;
- valori troppo piccoli possono farli diventare quasi nulli;
- in entrambi i casi il training può diventare instabile o molto lento.

Xavier sceglie la scala dei pesi in base a due quantità:

- `fan_in`: numero di ingressi del neurone;
- `fan_out`: numero di uscite del neurone.

In questo modo cerca di mantenere simile la varianza dei dati mentre attraversano i vari strati della rete.

## Distribuzione uniforme

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

## Distribuzione normale

Una variante usa una distribuzione normale con media zero e deviazione standard:

$$
\sigma = \sqrt{\frac{2}{fan\_in + fan\_out}}
$$

Quindi:

$$
W_{ij} \sim N(0, \sigma^2)
$$

Dove `N(0, sigma^2)` indica una distribuzione normale con media zero e varianza `sigma^2`.

## Esempio in C++

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

## Nel Transformer

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

## Limite pratico

Xavier è particolarmente adatta alle funzioni di attivazione simmetriche, come `tanh`. Con `ReLU` si usa spesso la **He initialization**, che tiene conto del fatto che ReLU annulla i valori negativi.

## Perché non usare direttamente scalari per i caratteri

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

### ID come indice, embedding come input

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

### Come scegliere la dimensione degli embedding

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

Per un vocabolario di circa 30 caratteri e un classificatore di parole semplice, si possono provare inizialmente dimensioni `5`, `8` e `16`. Se l'embedding è anche l'input del Transformer, spesso si imposta:

```text
embedding dimension = modelSize
```

così non serve una proiezione aggiuntiva. Per esempio, una configurazione didattica potrebbe essere:

```text
vocabolario: 30 caratteri
embedding dimension: 5
modelSize: 5
feedForwardSize: 10
```

La scelta finale dovrebbe essere la dimensione più piccola che raggiunge buone prestazioni sul validation set in modo stabile. In questo modo si limita il numero di parametri senza sacrificare la capacità del modello.

## Concetti fondamentali del training

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
