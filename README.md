CONTENTS OF THIS FILE
---------------------

* Status of this document.
This document is a submission to ECON 8070 Class by JAEYOUNG KIM. 
Comments on this document should be sent to jaeyouk@clemson.edu.

 * Introduction

 The project folder contains:
 (1) cleaning_ydummies.py
 (2) cleaning_no_stars.py
 (3) kfold_template.py 
 (4) runme.py
 (5) README
 (6) report.pdf
 (7) business_sample.json (not uploded)
 (8) business_no_stars.json
 (9) cleaned_data_ydummies.csv
 (10)cleaned_no_stars.csv


 * pre-requisite

 - Python


 * INSTRUCTIONS FOR EXECUTION

	(1) Run 'cleaning_ydummies.py'
	This file cleans the data file for training and test as the report.pdf specifies. The data file should be named as 'business_sample.json' or modified in the code directly. The cleaned dta is stored as a CSV file, named 'cleaned_data_ydummies.csv'.

	(2) Run 'cleaning_no_stars.py'
	This file cleans the data file for prediction as the same dimention and format of training and test data. The data file should be named as 'business_no_stars.json' or modified in the code directly. The cleaned dta is stored as a CSV file, named 'cleaned_no_stars.csv'.

	(3) Keep 'kfold_template.py' in the working directory
	The main analysis python code will import it to utilize KFold validation. 

	(4) Run 'runme.py
	This file trains the machine and predict the star rating using linear regression and random forest. Two models are compared in terms of R-squared. Accuracy scores and confusion matrices are computed for random forest as complementary metrics to compare other classification models. 

	(5) README

	(6) 'report.pdf' summarizes the procedure and results. 

	(7) 'business_sample.json': Example data set for training and test (not uploded due to the data size - unable to upload to Github)
 
 	(8) 'business_no_stars.json': Example data set for prediction

 	(9) cleaned_data_ydummies.csv: Example data set after cleaning for training and test
 	
 	(10)cleaned_no_stars.csv: Example data set after cleaning for prediction


 * TODO 
 Note that the R-squared from Random Forest model are negative, and the predicted star rating is out of range of the star rating in the training/test data set. Future study should clarify the validity of this analysis by rechecking the data and codes.

 * Troubleshooting
 Questions can be sent to jaeyouk@clemson.edu.
