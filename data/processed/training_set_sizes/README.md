## Effect of Training Set Size

This folder contains csvs of the metrics collected from the simple O'Reilly 
model when run on different training sets.

The number of training samples used for an experiment is in the file name. For
example, the file `10_models_25000_epochs_200_training_samples.csv` has 
metrics for 10 models trained on sets of 200 samples.

The general experimental setup for these are that in each metrics csv, there are
10 rows corresponding to 10 different models, where each is trained on N samples
for 25,000 epochs. Each of the models had a number of metrics collected at the
completion of every epoch, which corresponds to each of the columns of the csv.

For more information on the meaning of the metrics, check out `NB-0.3.1`.

*NOTE*: The contents of this directory are ignored by git, and is meant for 
local use.
