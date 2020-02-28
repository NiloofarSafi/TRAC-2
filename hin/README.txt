Train and Dev Dataset
=======================
This package contains two files
	- trac2_hin_train.csv 
	- trac2_hin_dev.csv 

'trac2_hin_train.csv' contains the train set for the shared task in Hindi. 
'trac2_hin_dev.csv' contains data for selecting and testing the model such that they do not overfit for the train set.

You can submit the final system which is trained on the total data taken from both the sets. You are also free to use a different subset as the dev data.

Both the files contain 4 columns in the following format
		unique_id,text,Sub-task A,Sub-task B

The columns are separated by comma and follows a minimal quoting pattern (such that only those columns are quoted which are in multiple lines or contain quotes in the text).


Test Dataset
=============
For testing, we will provide you test set at a later stage.

The test files will also be in csv format that will contain 2 columns in the following format (without the labels)
		unique_id,text

You will need to send us back the aggression level of the texts formatted as below
	unique_id,labels
	

Licence
========
The full dataset (both the train and test sets) is licensed under Creative Commons Non-Commercial Share-Alike 4.0 licence CC-BY-NC-SA 4.0
