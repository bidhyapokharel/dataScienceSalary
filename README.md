Data Science Salary Estimator: Project Overview

- Created a tool  that estimates data science salaries (MAE-$11K) to help data scientist negotiate 
their income when they get a job
- Scraped jobs from glassdoor using python and selenium
- Engineered features from the text of each job description to quantify the value companies put on Pyhton,excel,aws and spark
Optimize Linear, Lasso and Random Forest Regression using GridSearchCV to reach the best model.
- Built a client facing API using flask.


Code and Resources used: 

Python version: 3.8.5
Packages: Pandas, Numpy, Sklearn, Mattplotlib, Seaborn,Selenium, Flask, json, pickle
For Web Framework Requirements: pip install -r requirements.txt

Web Scraping:

Scraping was done from Glassdoor.

Data Cleaning:
- Parsed the numeric data as well as removed unnecessary data as per requirement

EDA:

I looked at the distributions of the data and the value counts for the various categorical values.

Model Building:

I used 3 different models:
1. Multiple Line Regression-
Baseline for the model.

2. Lasso Regression:
Because of the sparse data from the many categorical variables. I thought a normalized regression like lasso would be effective.

3. Random Forest:
Again, with the sparsity assiociated with the data. I thought that this would be a good fit.

Productionization:

In this step, I built a flask API endpoint that was hosted on a local website by following along with the TDS tutorial in the 
reference section above. The API endpoint takes in a request with a list of values fro 
a job listing and returns an estimated salary.
