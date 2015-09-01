# Classif_01092105


# Instructions
Ideally, you ll work with R or Python to carry out the following steps:

* o   Import the learning and text files
* o   Based on the learning file:
* o   Make a quick statistic based and univariate audit of the different columnsâ€™ content and produce the results in visual / graphic format.
* o   This audit should describe the variable distribution, the % of missing values, the extreme values, and so on.
* o   Create a model using these variables (you can use whichever variables you want, or even create you own; for example, you could find the ratio or relationship between different variables, the binarisation of â€œcategoricalâ€ variables, etc.) to modelize wining more or less than $50,000 / year. Here, the idea would be for you to test one or two algorithms, type regression logistics, or a decision tree. But, you are free to choose others if youâ€™d rather.
* o   Choose the model that appears to have the highest performance based on a comparison between reality (the 42nd variable) and the modelâ€™s prediction.
* o   Apply your model to the test file and measure itâ€™s real performance on it (same method as above).
 
# Requirements
I am using ANACONDA python distribution : 

* python 2.7
* numpy 1.9.2
* matplotlib 1.4.3
* scikit learn 1.16.1

I have added fuzzywuzzy to the distribution (https://github.com/seatgeek/fuzzywuzzy)

# IMPORT METADATAS

First I wanted to add column names on top of each column. Not wanting to modify the metadata file (census_income_metadata.txt). I have struggled quite a bit with this initial task. 

I have faced problems such as discrepancy between informations, peace of informations along the file, differently orthographed descriptions, unused column...

I have created a ParseMetadata class to which you give the filename you want to parse. 
Informations such as line number are hard coded for now.

Headers are modified manually for the most problematic ones.

Then I am using fuzzywuzzy in order to have the right ACRONYMS in the right order to fit the 42 columns

# IMPORT LEARNING / TEST FILE 

Files are imported using pandas.read_csv.
I add the column names (Acronyms) on top of each column 

For each column a report is printed (These reports can be desactivate  putting the REPORTS parameter to 0)and an histogramm /bar plot depending on their dtype. 

In histogramms the red vertical line shows the mean value. 

These plottings can be desactivate putting the SHOW_PLOTS_UNIVARIATE parameter to 0

# MODELS

I have tested several models : "Decision Tree","Gradient Boosting","Random Forest", "AdaBoost", "Naive Bayes" and studied their learning curve depending on the size of the data. 

From now on I have kept GradientBoostingClassifier and AdaBoost, using validation curve to study the evolution of performances according to different parameters.

# INSIGHT

The fact of making more than $50 000 per year depends on the 
level of education (20%) then the choice in the major occupation(12%), 
your dividends from stocks (6%), the number of week worked in the year, even the marital status(4%) play a role. 





