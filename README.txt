David Vinegar dvinegar3

To obtain training error:
1)	Open Weka.  
2)	Click on the Explorer button.
3)	On the preprocess tab, click the open file button and open the desired training set from the “TrainingSets folder”.  Each file in the folder has its dataset as a prefix (either “income” or “Housing” and the percent of the total set used in the specific set as a percent.  For example, 10% would mean that 10% of the training set is contained wthin the set.)
4)	Click on the classify tab.  
5)	Choose the appropriate classifier (Classifiers used: Function.SMO, lazy.IBk, AdaboostM1, Trees.J48, functions.MultiLayerPerceptron) 
6)	Under Test options, choose:  Use training set
7)	Click Start
8)	Train error rate will be the percent of incorrectly classified instances.

To obtain testing error:
1)	Repeat steps 1-5 from “obtaining training error”.
2)	Under Test options, choose “Supplied test set”.
3)	Click on the “Set…” button.  
4)	Choose the desired test set from the TestSets folder.
5)	Click Start
6)	Test error rate will be the percent of incorrectly classified instances.

To obtain Cross validation error:
1)	Complete steps 1 and 2) of “To obtain training error”.  
2)	On the preprocess tab, click the open file button and open the desired cross validation set to train with from the “CrossValidationSets” folder.  Each file in the folder has its dataset as a prefix (either “income” or “Housing” and the percent of the total set used in the specific set as a percent.  For example, 1% would mean that 10% of the training set is contained within the set.)
3)	Click on the classify tab.  
4)	Choose the appropriate classifier (Classifiers used: Function.SMO, lazy.IBk, AdaboostM1, Trees.J48, functions.MultiLayerPerceptron) 
5)	Under Test options, choose “Supplied test set”.
6)	Click on the “Set…” button.  
7)	Choose the desired test set from the CrossValidationSets folder.  The two cross validation sets intended to be tested on are labeled as housing_cvTEST_from_CVtrain.arff and
income_cvTEST_from_CVtrain.arff
8)	Click Start
9)	Cross validation error rate will be the percent of incorrectly classified instances.


Notes: Parameter tweaking was completed by changing the desired parameter on the classifier tab in Weka.  Clicking on the  text box for the classifier chosen will cause a menu to pop up containing the various parameters available to modify. 

To obtain variance:
1) 	In MS Excel, subtract cross validation error from training error, and square the difference.
2) 	Repeat step 1 at each level of data inclusion, for example, 10% of data included through 70% of data included. 
3) 	For the set of differences created by step 2, use the MS Excel “Var” function on that
data set. 
