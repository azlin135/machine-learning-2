# machine-learning-2

## Classification using KNN and LDA

Data used -> newthyroid.txt <br />
Information about data -> This data contain measurements for normal patients and those with hyperthyroidism. The first variable class=n if a patient is normal and class=h if a patients suffers from hyperthyroidism. The remaining variables feature1 to feature5 are miscellaneous medical test measurements.

## Steps undertaken:

• Applied kNN and LDA to classify the newthyroid.txt data: random split of the data to a training set (70%) and a test set (30%) and repeated the random split 20 times. <br />
For kNN, repeated 5-fold cross-validation five times to choose k from (3, 5, 7, 9, 11, 13, 15, 17, 19, 21). Used AUC as the metric to choose k, i.e. chose k with the largest AUC. <br />
Stored the 20 AUC values of kNN and LDA in two vectors. <br />
• For the first random split, the ROC curves of kNN and LDA are plotted on the same plot. <br />
• Two boxplots for the 20 AUC values of kNN and LDA are represented on one plot.
