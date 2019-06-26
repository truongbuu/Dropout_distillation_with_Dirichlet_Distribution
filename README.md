# MC_Dropout_distillation_with_Dirichlet_Distribution

This project attempts to estimate the epistemic uncertainty of a neural network trained by MC-dropout in real-time.
Suppose that we already have a classifier trained with dropout, by turning on the dropout layer at test time, the network gives a distribution of probability vector for each input. This distribution is then used to calculate the so-called epistemic and aleatoric uncertainty.
Since this sampling process (known as MC-dropout) requires about 50~100 samples/input to estimate these two uncertainties, it is not applicable for real-time applications.

One way to approach this problem is to learn the relation between the input and output distribution of that classifier (called it a teacher-network). We train a student network to learn this relationship. The distribution comes from the student network should have a known closed-form such that we can estimate the epistemic uncertainty (by taking the sum of variance of the distribution), predictive uncertainty ( taking the entropy of the mean probability vector) and aleatoric uncertainty (by substracting the predictive uncertainty by the epistemic one). Dirichlet distribution is one of the distribution known for outputing probability vector samples. It is parametrized by the \alpha_i, for i = 0, 1, ..., Number of Classes.  
