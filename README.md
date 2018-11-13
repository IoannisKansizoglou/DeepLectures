

# Βαθιά Μάθηση - Deep Learning <img src='https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/logoDUTH.png' width='30%' align='right'>

**Laboratory of Robotics & Autonomous Systems**<br />
_Kansizoglou Ioannis, PhD Candidate_<br />
<i> ikansizo@pme.duth.gr </i>



### Τι είναι ένα (τεχνητό) νευρωνικό δίκτυο ?



<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/gifs/nural-network_3.gif" width="500" align='centre'>



- Είναι ένα δίκτυο από απλούς υπολογιστικούς κόμβους (**νευρώνες**), διασυνδεδεμένους μεταξύ τους.
- Αξιοποιεί μία **απλοποιημένη** μορφή του κεντρικού νευρικού συστήματος (εγκεφάλου).
- Αποτελεί μέθοδο **μηχανικής μάθησης**.



### Δομή νευρώνα: μια βαθύτερη ματιά

Κάθε νευρώνας επιστρέφει **μία τιμή**.

<p style='text-align: center'>z = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + ... + w<sub>n</sub>x<sub>n</sub> + b</p>

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/Neuron.jpg" width="500">



##### Η συνάρτηση ενεργοποίησης f (activation function)

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/ActivationFunctions.png" width="500">

Στα κρυφά επίπεδα πλέον χρησιμοποιείται σχεδόν αποκλειστικά η **ReLU**.

Συνεπώς, ολοκληρωμένα η έξοδος του νευρώνα είναι:  

<p style='text-align: center'><b>y = ReLU(w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + ... + w<sub>n</sub>x<sub>n</sub>+b)</b></p>



### Μηχανική μάθηση

Πώς μαθαίνει ο άνθρωπος ?? =======> Μέσω **εμπειριών !!**

_Ορισμός Μηχανικής Μάθησης:_ Κλάδος που ερευνά μεθόδους για την εκπαίδευση μίας μηχανής ούτως ώστε μέσω εμπειριών (δειγμάτων) να μάθει να διεκπεραιώνει μία συγκεκριμένη εργασία με ορισμένο βαθμό απόδοσης.

Πρακτικά, εκπαιδεύουμε τις παραμέτρους ***w*** μίας συνάρτησης ***F***, με σκοπό όταν δέχεται μια είσοδο ***x*** να παράγει την επιθυμητή έξοδο ***y***. Μαθηματικά μπορούμε να πούμε ότι θέλουμε:
$$
\begin{align*}
y = F(x;w) \approx label
\end{align*}
$$


##### Έστω ότι θέλουμε να εκπαιδεύσουμε ένα νευρώνα με συνάρτηση F(x) = σ(w<sub>0</sub> + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>) για να κατηγοριοποιήσουμε τα δεδομένα της παρακάτω εικόνας:

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_6_0.png" width="50%">



Παρακολουθούμε την τιμή **z = w<sub>0</sub> + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>**:

- Αν z>0 τότε η F>0.5 και ενεργοποιείται ο νευρώνας (κλάση 1)
- Αν z<0 τότε η F<0.5 και δεν ενεργοποιείται ο νευρώνας (κλάση 0)

Τα βήματα της εκπαίδευσης έχουν ως εξής:



#### 1. ΟΡΓΑΝΩΝΟΥΜΕ ΤΑ ΔΕΙΓΜΑΤΑ ΕΚΠΑΙΔΕΥΣΗΣ

Είναι σημαντικό να ανακατεύουμε τα δεδομένα.

```
[x1,x2,label]
[[2 4 1]
 [5 2 0]
 [1 2 1]
 [4 3 0]
 [7 4 0]
 [5 6 1]
 [3 6 1]
 [4 5 1]
 [7 8 1]
 [8 6 0]
 [6 5 0]
 [2 1 0]]
```



#### 2. ΑΡΧΙΚΟΠΟΙΟΥΜΕ ΤΙΣ ΠΑΡΑΜΕΤΡΟΥΣ (**w**)

Κακή αρχικοποίηση καθυστερεί, μειώνει την απόδοση ή καθιστά αδύνατη την σύγκλιση.

```
[ w0 ,w1 ,w2 ] = [-8, 1, 1.4]
```

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_10_1.png" width="50%">



#### 3. ΕΙΣΑΓΟΥΜΕ ΤΟ ΠΡΩΤΟ ΔΕΙΓΜΑ

Υλοποιούμε την πράξη: w<sub>0</sub> + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> 

Αν το σημείο είναι πάνω από τη γραμμή του νευρώνα το αποτέλεσμα είναι θετικό, διαφορετικά προκύπτει αρνητικό. Στην οριακή περίπτωση που ανήκει στην γραμμή το αποτέλεσμα είναι μηδενικό. 

```
Δείγμα 0: [2 4 1]
```



<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_12_1.png" width="50%">

```
z = -0.40000000000000036 και η έξοδος του νευρώνα: y = 0.40131233988754794
Συνεπώς, ο νευρώνας κατηγοριοποιεί το δείγμα 0 στην κόκκινη κλάση.
Λανθασμένη υπόδειξη!
```



#### 4. ΥΠΟΛΟΓΙΖΟΥΜΕ ΤΟ ΣΦΑΛΜΑ

Εισάγουμε την έξοδο του νευρώνα, αλλά και την πραγματική τιμή του δείγματος σε μία συνάρτηση κόστους. Η πιο απλή είναι η τετραγωνική διαφορά:
$$
\begin{align*}
J(y)=\frac{1}{2}(y-label)^2
\end{align*}
$$
Θέλουμε η έξοδος να προσεγγίζει τις τιμές 0,1 όχι απλά να προβλέπει τις κλάσεις. Στην περίπτωσή μας έχουμε:

```
Σφάλμα για το δείγμα 0:   loss = 0.17921345718546142
```



#### 5. ΕΠΑΝΑΛΑΜΒΑΝΟΥΜΕ ΤΑ ΒΗΜΑΤΑ 3 & 4 ΓΙΑ ΟΛΑ ΤΑ ΔΕΙΓΜΑΤΑ

Αθροίζουμε τα σφάλματα και υπολογίζουμε το συνολικό σφάλμα του συστήματος

```
Δείγμα 0: [2 4 1]
Έξοδος νευρώνα: y = 0.40131. Λανθασμένη υπόδειξη!
Σφάλμα: loss = 0.1792134572
Δείγμα 1: [5 2 0]
Έξοδος νευρώνα: y = 0.45017. Σωστή υπόδειξη!
Σφάλμα: loss = 0.101324715
Δείγμα 2: [1 2 1]
Έξοδος νευρώνα: y = 0.01477. Λανθασμένη υπόδειξη!
Σφάλμα: loss = 0.4853351043
Δείγμα 3: [4 3 0]
Έξοδος νευρώνα: y = 0.54983. Λανθασμένη υπόδειξη!
Σφάλμα: loss = 0.1511587123
Δείγμα 4: [7 4 0]
Έξοδος νευρώνα: y = 0.99005. Λανθασμένη υπόδειξη!
Σφάλμα: loss = 0.4900977173
Δείγμα 5: [5 6 1]
Έξοδος νευρώνα: y = 0.9955. Σωστή υπόδειξη!
Σφάλμα: loss = 1.01082e-05
Δείγμα 6: [3 6 1]
Έξοδος νευρώνα: y = 0.9677. Σωστή υπόδειξη!
Σφάλμα: loss = 0.0005214985
Δείγμα 7: [4 5 1]
Έξοδος νευρώνα: y = 0.95257. Σωστή υπόδειξη!
Σφάλμα: loss = 0.0011246067
Δείγμα 8: [7 8 1]
Έξοδος νευρώνα: y = 0.99996. Σωστή υπόδειξη!
Σφάλμα: loss = 7e-10
Δείγμα 9: [8 6 0]
Έξοδος νευρώνα: y = 0.99978. Λανθασμένη υπόδειξη!
Σφάλμα: loss = 0.4997752085
Δείγμα 10: [6 5 0]
Έξοδος νευρώνα: y = 0.99331. Λανθασμένη υπόδειξη!
Σφάλμα: loss = 0.4933295462
Δείγμα 11: [2 1 0]
Έξοδος νευρώνα: y = 0.00995. Σωστή υπόδειξη!
Σφάλμα: loss = 4.95192e-05

Συνολικό Σφάλμα: 2.4019401941538843
```



#### 6. ΔΙΟΡΘΩΝΟΥΜΕ ΤΙΣ ΠΑΡΑΜΕΤΡΟΥΣ ΜΑΣ ΜΕΣΗ ΤΗΣ ΜΕΘΟΔΟΥ: **GRADIENT DESCENT**

| 2D                                                           | 3D                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/LossFunction.png" width="100%"> | <img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/LossFunction3D.png" width="100%"> |



Τα βάρη διορθώνονται μέσω της:
$$
\begin{align*}
w_i = w_i - \alpha\frac{\partial J}{\partial w_i}
\end{align*}
$$


Συνεπώς, υπολογίζουμε μερικές παραγώγους της συνάρτησης κόστους. 

```
[w0,w1,w2] = [-8, 1, 1.4]
[w0',w1',w2'] = [-8.010284377556651, 0.9102333729710631, 1.391632618150986]
```

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_18_1.png" width="50%">



#### ΑΣ ΕΚΤΕΛΕΣΟΥΜΕ ΟΛΟΚΛΗΡΩΜΕΝΑ ΤΗΝ ΕΚΠΑΙΔΕΥΣΗ

Κάθε επανάληψη, η οποία ορίζεται από την εισαγωγή του πρώτου μέχρι και του τελευταίου δείγματος εκπαίδευσης συμπεριλαμβάνοντας και την εφαρμογή του αλγορίθμου Gradient Descent, ονομάζεται **epoch**.

Εκτελούμε την εκπαίδευση για προκαθορισμένο αριθμό epochs, είτε επαναλαμβάνουμε μέχρι το σφάλμα εκπαίδευσης να ελλατωθεί κάτω από ένα επιθυμητό επίπεδο.

Έστω ότι θέλουμε το σφάλμα να γίνει μικρότερο από 0.01

```
Epoch_0 Συνολικό Σφάλμα: 2.4019401941538843

Epoch_1 Συνολικό Σφάλμα: 2.329520702681356

Epoch_2 Συνολικό Σφάλμα: 2.284788997951755

Epoch_3 Συνολικό Σφάλμα: 2.2504521846307064

Epoch_4 Συνολικό Σφάλμα: 2.218260766279002

Epoch_5 Συνολικό Σφάλμα: 2.1825972045368642

Epoch_6 Συνολικό Σφάλμα: 2.135130764910242

Epoch_7 Συνολικό Σφάλμα: 2.0569776607674903

...

Epoch_685 Συνολικό Σφάλμα: 0.010042452159097071

Epoch_686 Συνολικό Σφάλμα: 0.010016644423407106

Epoch_687 Συνολικό Σφάλμα: 0.009990960773269853
```



|              Μετατόπιση της γραμμής του νευρώνα              |             Συνολικό σφάλμα κατά την εκπαίδευση              |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_20_1.png" width="100%"> | <img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_20_2.png" width="100%"> |



Και το αποτέλεσμα είναι...

```
[ w0 ,w1 ,w2 ] = [-4.7664440430506545, -4.193920379177114, 5.452505839107012]
```

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_22_1.png" width="50%">



### Τι σχέση έχουν όλα αυτά με εμάς...



Συνελικτικά νερωνικά δίκτυα (Convolutional Neural Networks - CNNs)

- Με την χρήση **φίλτρων** και την πράξη της συνέλιξης εντοπίζουμε χρήσιμες πληροφορίες (features maps) απευθείας πάνω στην εικόνα.

```
image matrix size:  (576, 768)
```

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_25_1.png" width="50%">

Για παράδειγμα εφαρμόζουμε **Edge Detection** για να τονίσουμε τις **κάθετες** ακμές της εικόνας.

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/output_27_0.png" width="100%">

Τι γίνεται όμως στην περίπτωση που δεν γνωρίζουμε τι είδους ακμές πρέπει να εντοπιστούν για να επιλυθεί ένα πρόβλημα?

- ***ΛΥΣΗ:*** Εκπαιδεύουμε τα φίλτρα για να αποφασίσουνε μόνα τους να εντοπίζουν τα κατάλληλα στοιχεία στην εικόνα.



### CONVOLUTIONAL LAYER

Σε ένα ConvLayer μπορούμε να φανταστούμε τους νευρώνες να έχουν την παρακάτω διάταξη.

| A                                                            | B                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/tikz44.png" width="100%"> | <img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/tikz45.png" width="100%"> |



Η τιμή κάθε νευρώνα σε αυτά τα επίπεδα προκύπτει από μία τετραγωνική **περιοχή** των νευρώνων του προηγούμενου επιπέδου. Περιέχει δηλαδή χωρική πληροφορία, που χρειάζεται στις εικόνες.

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/convolve.png" width="70%">



Η περιοχή αυτή συνελίσσεται με τα φίλτρα του επιπέδου (μπλε τετράγωνο) που περιέχουν τις εκπαιδευόμενες παραμέτρους και παράγουν ορισμένους χάρτες χαρακτηριστικών (***feature maps***). Το μέγεθος του φίλτρου (K) καθορίζει και το μέγεθος της περιοχής.

Στην παραπάνω εικόνα έχουμε σαν είσοδο εικόνα ενός καναλιού π.χ. _grayscale_. Για _RGB_ εικόνες χρειαζόμαστε φίλτρο τριών καναλιών ένα για κάθε χρώμα, όπως φαίνεται παρακάτω:



<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/gifs/rgb.gif" width="100%">



<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/conv.png" width="60%">



Επίσης θέλουμε να αναζητούμε όχι ένα αλλά περισσότερα στοιχεία σε μια εικόνα. Συνεπώς χρησιμοποιούμε πολλαπλά φίλτρα που παράγουν διαφορετικούς feature maps. Τοποθετώντας τους τον ένα πίσω από τον άλλο ορίζουμε το βάθος (**depth**) του επιπέδου.

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/conv2.png" width="60%">



#### Βασικά χαρακτηριστικά συνέλιξης εικόνας:

- **Stride (S):** Ο αριθμός των pixels που μετατοπίζεται το φίλτρο στον οριζόντιο ή τον κατακόρυφο άξονα. Παρακάτω S=2 ή [2,2].
- **Padding (P):** Ο αριθμός των pixels που προσθέτουμε γύρω-γύρω από την είσοδο στην κάθε διεύθυνση. Παρακάτω P=1 ή [1,1].

<img src="https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/gifs/stride2.gif" width="100%">

Το μέγεθος του feature map έχει 3 διαστάσεις [ $ width\times height\times depth $ ] και αντιστοιχεί στον **αριθμό των νευρώνων του επιπέδου**. Οι διαστάσεις καθορίζονται ως:


$$
\begin{align*}
width = \frac{I_w-K_w+2P}{S}+1
\end{align*}
$$

$$
\begin{align*}
height = \frac{I_h-K_h+2P}{S}+1
\end{align*}
$$

$$
\begin{align*}
depth = Number\_of\_filters
\end{align*}
$$



Αντίστοιχα οι διαστάσεις των παραμέτρων του επιπέδου έχουν ως εξής:


$$
\begin{align}
weights = [ K_w \times K_h \times channels \times depth ]
\end{align}
$$

$$
\begin{align}
bias = [ depth ]
\end{align}
$$



***ΠΡΟΣΟΧΗ***: Τα κανάλια (channels) καθορίζονται από τον αριθμό φίλτρων (άρα το depth) του προηγούμενου επιπέδου. 



### POOLING LAYER

Σε ένα PoolLayer απλά κάνουμε μείωση μεγέθους (**downsample**) της εικόνας. Βασικές μέθοδοι pooling είναι: max, min, average. Συνήθως, στα δίκτυα εφαρμόζουμε **max pool**.

***Max Pool:*** Ένα φίλτρο μεγέθους [$K_w \times K_h$]  σαρώνει **ανά μονάδα βάθους (depthwise)** το εισαγόμενο επίπεδο και διατηρεί μόνο την μέγιστη τιμή της κάθε περιοχής.



<img src='https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/max-pooling.png' width='70%'>



Παρόμοια έχουμε:

- **Stride (S)**: Συνήθως S=K ή [$K_w,K_h$] 
- **Padding (P)**: Συνήθως P=0 ή [0,0]

Υπό αυτές τις συνθήκες στην έξοδο προκύπτει feature map μεγέθους: 


$$
[ (\frac{I_w-K_w}{S}+1) \times (\frac{I_h-K_h}{S}+1) \times depth ] = [ \frac{I_w}{K_w} \times \frac{I_h}{K_h} \times depth ]
$$


Για παράδειγμα η παραπάνω εικόνα από $ [ 4\times4\times1 ] $ συμπιέστηκε σε $ [ 2\times2\times1 ] $ pixels. Η μορφή του στις 3 διαστάσεις έχει ως εξής:

<img src='https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/pool2.png' width='70%'>



***ΠΡΟΣΟΧΗ***: Το PoolLayer **δεν** εχει εκπαιδευόμενες παραμέτρους. Μειώνει το μέγεθος του επιπέδου της εισόδου και διατηρεί το βάθος της.



### AlexNet

 Σχεδιάστηκε από τον _Alex Krizhevsky_ και εκπαιδεύτηκε στο _ImageNet_.

<img src='https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/AlexNet.png' width='90%'>



_Paper:_ Imagenet classification with deep convolutional neural networks,
_Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E, Advances in neural information processing systems, 2012_



<img src='https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/imagenet.jpeg' width='40%'>

~ 14 εκατομμύρια εικόνες

- 1.2 Μ, εικόνες εκπαίδευσης (training)
- 100 K, εικόνες επαλήθευσης (test)
- 1000 κλάσεις

<img src='https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/imagenet2.png' width='80%' align='center'>



### Και τι υπάρχει μέσα σε ένα CNN ?



<img src='https://raw.githubusercontent.com/IoannisKansizoglou/DeepLectures/master/images/kernels.jpeg' width='80%'>



<table>
<tr>
    <th align='center'> Size/Operation </th> 
    <th align='center'> Filter </th>
    <th align='center'> Stride </th>
    <th align='center'> Padding </th>
</tr>
<tr>
    <td align='center'> 227x227x3 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>Conv1+ReLU</i> </td>
    <td align='center'> 11x11x3x96 </td>
    <td align='center'> 4 </td>
    <td align='center'> 0 </td>
</tr>
<tr>
    <td align='center'> 55x55x96 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>Max Pool</i> </td>
    <td align='center'> 3x3x96 </td>
    <td align='center'> 2 </td>
    <td align='center'> 0 </td>
</tr>
<tr>
    <td align='center'> 27x27x96 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>Conv2+ReLU</i> </td>
    <td align='center'> 5x5x96x256 </td>
    <td align='center'> 1 </td>
    <td align='center'> 2 </td>
</tr>
<tr>
    <td align='center'> 27x27x256 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>Max Pool</i> </td>
    <td align='center'> 3x3x256 </td>
    <td align='center'> 2 </td>
    <td align='center'> 0 </td>
</tr>
<tr>
    <td align='center'> 13x13x256 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>Conv3+ReLU</i> </td>
    <td align='center'> 3x3x256x384 </td>
    <td align='center'> 1 </td>
    <td align='center'> 1 </td>
</tr>
<tr>
    <td align='center'> 13x13x384 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>Conv4+ReLU</i> </td>
    <td align='center'> 3x3x384x384 </td>
    <td align='center'> 1 </td>
    <td align='center'> 1 </td>
</tr>
<tr>
    <td align='center'> 13x13x384 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>Conv5+ReLU</i> </td>
    <td align='center'> 3x3x384x256 </td>
    <td align='center'> 1 </td>
    <td align='center'> 1 </td>
</tr>
<tr>
    <td align='center'> 13x13x256 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>Max Pool</i> </td>
    <td align='center'> 3x3x256 </td>
    <td align='center'> 2 </td>
    <td align='center'> 0 </td>
</tr>
<tr>
    <td align='center'> 6x6x256 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>FC6+ReLU</i> </td>
    <td align='center'> 1x1x4096 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> 4096 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>FC7+ReLU</i> </td>
    <td align='center'> 1x1x4096 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> 4096 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> <i>FC8+ReLU</i> </td>
    <td align='center'> 1x1x4096 </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
<tr>
    <td align='center'> 1000 <i>classes</i> </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
    <td align='center'> - </td>
</tr>
</table>



### References - Sites

1] http://neuralnetworksanddeeplearning.com/

2] http://cs231n.github.io/convolutional-networks/

3] https://anhvnn.wordpress.com/

4] http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html

5] https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2

6] https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637

7] http://www.image-net.org/

