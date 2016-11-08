TL;DR - For an analysis of 6 unsupervised learning techniques, please see dvinegar3-analysis.pdf. Code to justify the analysis written is in main.py. 

First students were asked to run a number of experiments:

Experiment 1:  Run clustering algorithms K-means Expectation Maximization on two datasets and describe what you see.
Experiment 2: Run 4 dimensionality reduction algorithms (PCA, ICA, Randomized Projection, and Feature Selection algorithm of student's choice) on the two datasets and describe what you see.  
Experiment 3: Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
Experiment 4: Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
Experiment 5: Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms (you've probably already done this), treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.

Then, students were expected to analyze results.  The analysis includes:

a discussion of your datasets, and why they're interesting: If you're using the same datasets as before at least briefly remind us of what they are so we don't have to revisit your old assignment write-up.
explanations of your methods: How did you choose k?
a description of the kind of clusters that you got.
analyses of your results. Why did you get the clusters you did? Do they make "sense"? If you used data that already had labels (for example data from a classification problem from assignment #1) did the clusters line up with the labels? Do they otherwise line up naturally? Why or why not? Compare and contrast the different algorithms. What sort of changes might you make to each of those algorithms to improve performance? How much performance was due to the problems you chose? Be creative and think of as many questions you can, and as many answers as you can. Take care to justify your analysis with data explictly.
Can you describe how the data look in the new spaces you created with the various aglorithms? For PCA, what is the distribution of eigenvalues? For ICA, how kurtotic are the distributions? Do the projection axes for ICA seem to capture anything "meaningful"? Assuming you only generate k projections (i.e., you do dimensionality reduction), how well is the data reconstructed by the randomized projections? PCA? How much variation did you get when you re-ran your RP several times (I know I don't have to mention that you might want to run RP many times to see what happens, but I hope you forgive me)?
When you reproduced your clustering experiments on the datasets projected onto the new spaces created by ICA, PCA and RP, did you get the same clusters as before? Different clusters? Why? Why not?
When you re-ran your neural network algorithms were there any differences in performance? Speed? Anything at all?

To run the code used in the analysis, follow the instructions below.  

1) Open up main.py in the SupportingFiles directory.
2) Change the two filepath variables "incomeDataFilePath" and "housingDataFilePath" to their corresponding location on your hard drive using the SupportingFiles 
“housing_training_70.csv” and “income_training.cv”.
3) Change the file path variable “incometestDataFilePath” (should be in two places in the code) to their corresponding location where “income_test.csv” is.
4) Run main.py
