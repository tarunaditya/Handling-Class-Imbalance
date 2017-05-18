# Handling-Class-Imbalance
Often in classification problems have class imbalance eg problem of finding out propensity to click (CTR for marketing campaigns dont go beyond 9%) / Impressions viewed in a given spot. The typical classifiers when trained on such a data simply classifies a new training data point to either click or no click. In other words they would be having poor Precision rates/ specificity. There are a few ways to have a work around, one way is to introduce more hypothetical datapoints of the infrequent class, second way is to sample from more frequent class 'smartly', thirdly look at the code & its comments to understand more about certain new packages in R that i found interesting that does much better job with respect to AUC metrics
