# Combinatorial Generalizaton and Interactivity

The task described here is meant to be an implementation of O'Reily's task in
[Generalization in Interactive Networks: The Benefits of Inhibitory Competition and Hebbian Learning](https://www.mitpressjournals.org/doi/10.1162/08997660152002834).
The majority of the text below is taken directly from section **2** of the
paper.

General Properties
------------------

- Combinatorial structure is implemented by having *four* different input-output
  slots.
- The output mapping for a given slot depends only on the corresponding input
  pattern for that slot (see
  [Brousse, 1993](https://scholar.colorado.edu/csci_techreports/647/);
  [Noelle & Cottrell, 1996](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.2295),
  for similar tasks).
- Each slow has a vocabulary of input-output mappings.
- Input vocabulary consists of all 45 combinations of 5 horizontal and 5
  vertical bars in a 5x5 grid.
- Output mapping is a localist identification of the two input bars (similar to
  bar tasks used by
  [Foldiak, 1990](https://link.springer.com/article/10.1007%2FBF02331346);
  [Saund, 1995](https://www.mitpressjournals.org/doi/10.1162/neco.1995.7.1.51);
  [Zemel, 1993](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.6050);
  [Dayan & Zemel, 1995](http://www.gatsby.ucl.ac.uk/~dayan/papers/cdz95.pdf)\).
- Total number of distinct input patterns is approximately 4.1 million.
- Models are intended only to train on 100 randomly constructed examples, and
  then test on an arbitraily large testing set (500 in the paper).
- Error criterion is scored such that each output unit has to be on the right
  side of 0.5 according to the correct target pattern.

Desiderata
----------

The paper described the task as having several desiderata.

- It has a simple combinatorial structure that allows for novel inputs to be
  composed from a small vocabulary of features.
- There is some interesting substructure to the vocabulary mapping at each slot.
- The structure of the task should be apparent in the weight patterns of the
  models.

Paper Links
-----------

- [Brousse, O. (1993). Generativity and systematicity in neural network combinatorial learning](https://scholar.colorado.edu/csci_techreports/647/)
- [Noelle, D. C., & Cottrell, G. W. (1996). In search of articulated attractors](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.2295)
- [F¨oldi´ak, P. (1990). Forming sparse representations by local anti-Hebbian learning](https://link.springer.com/article/10.1007%2FBF02331346)
- [Saund, E. (1995). A multiple cause mixture model for unsupervised learning](https://www.mitpressjournals.org/doi/10.1162/neco.1995.7.1.51)
- [Zemel, R. S. (1993). A minimum description length framework for unsupervised learning](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.53.6050)
- [Dayan, P., & Zemel, R. S. (1995). Competition and multiple cause models](http://www.gatsby.ucl.ac.uk/~dayan/papers/cdz95.pdf)
