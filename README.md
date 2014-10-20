#Allstate Prediction Challenge

The task for this Kaggle competition is to predict the purchased coverage options using a limited subset of the total interaction history. If this is accomplished the quoting process can be shortened and the issuer has a higher probability of keeping the customer's business.

A description of the feature data can be found [here.](https://www.kaggle.com/c/allstate-purchase-prediction-challenge/data)

Dependancies: matplotlib, PANDAS, sci-kit learn, SciPy, Numpy

##Exploratory Analysis

###Initial Distribution Analysis

I generated over a dozen scatter plot matrices and histograms to try to visualize any relationships between
the feature data and the targeted labels. In the scatter matrices, all points are labeled by color which
indicate which policy option the points are associated with. The histograms are similiarly labeled and colored.
All of these mentioned plots have file names beginning with the letter of the policy option 
(like [B_label_hist](https://github.com/alexkcode/allstate_challenge/blob/master/B_label_hist.png)). There were no obvious relationships, but the continuous features
, like car_age crossed with cost, tended to show a little more seperation than the discrete features. Looking
at the histograms, there were again no obvious seperations between distributions for the different labels.
Note, when looking at the histograms, the spike at the end of the plots for the ages of people (e.g. age_oldest)
occurs most likely because of binning, either from Numpy, or from Allstate when they collected the data. Most
likely it is a catch all bin for all people older than a certain number (around 75 it looks like).

###Feature Change Analysis

Most of the visualization and explorative analysis is generated in the [visuals.py](https://github.com/alexkcode/allstate_challenge/blob/master/visuals.py) 
script and in IPython. The script functions as a summary of the important 
parts of my IPython explorations. 

![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/feature_changes.png "feature_changes.png")

The first analysis I did was look at the changes in the customers and the 
policies they view in (shown in plot above). I notice a couple things:
* There isn't a lot of change in attributes of the customers between the 
first shopping point and the purchase point. Although it may be intuitive 
that people would not change where they lived or where they looked at the 
policies dramatically, within the days that look at the different insurance policies. 
* The more obvious changes show up in the cost of the plans they look at 
between the first and last shopping point and the actual choices for the 
policy options. 

Why look at these changes? Since this information is eventually going to be fed
into a machine learning algorithm I want to insure that the data is representative, 
especially since the original data is rather unbalanced in terms of the weight of 
the data from each customer. This could introduce unnecessary variance in the model.
Since we only care about what policy plan the customers actually purchase anyway, 
dropping all data that isn't directly connected to the actual purchase seems fair. 

###Viewed Policy Analysis

![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/Policy_Changes_from_First_to_Last.png "Policy_Changes_from_First_to_Last.png")
![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/Policy_Changes_from_First_to_Last_by_Degree.png "Policy_Changes_from_First_to_Last_by_Degree.png")
![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/Second_to_Last_Policy_Changes.png "Second_to_last_policy_changes.png")
![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/Second_to_Last_Policy_Changes_by_Degree.png "Second_to_last_policy_changes_by_Degree.png")

Now, let's look at the policy changes more closely, specifically between 
the first and last shopping points, and the second to last shopping point 
and the last shopping point, as in the above plots. It's clear from the above
plots that customers narrrow down their choices as they near their purchase point, shown
by the narrowing of the distribution of the degree of difference between shopping points.

The probability of a customer purchasing his/her last view insurance policy scheme
is 64.33%. This was calculated by taking the areas of the tails of the histogram,
excluding the rectangle above 0.

##Challenges

![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/missings.png "missings.png")

* Clearly, from above, something needs to be done about the 
risk_factor column if it's to be used for in the prediction model.
* Can the second to last viewed policy be utiilized to help predict 
the policy the customer actually purchases?

##Model Analysis

![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/clean0/LogisticRegressionOVR_A.png)
![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/clean0/LinearSupportVectorOVR_A.png)
![alt text](https://github.com/alexkcode/allstate_challenge/blob/master/clean0/RandomForestClassifier_A.png)

The scores of all the models I used were very similiar across all the policy options. The best performances 
were usually for policy options A and D with a mean accuracy around 70% (for cross validation). Both 
logistic regression and linear support vector machines had high bias issues while random forest
had a high variance issue. The code for this section can be found in [model.py] and a sample of 
the plots are found above. For this particular part I cleaned missing data by simply removing any rows
that had NAs.

##Lessons Learned

###Visualizations

* Sometimes simple is best, especially for complex data. 
  * Simple plots can communicate more information more quickly.
* Pay careful attention to binning, it can really affect how data is represented.
* Sampling is important to pay attention to as well.
  * Oversampling can lead to cluttered plots.
  * Undersampling may misrepresent data.
  * Sampling correctly can solve these problems and save time as well.

###Tools

* Pick a tool and stick with it. And also throughly investigate it.
  * Using multiple tools (or packages in my case) can give versitality.
  * But they can also add unnecessary programming complexity.
    * In my case I jumped between Numpy and PANDAS unnecessarily, 
    PANDAS already has a lot of built in functionality. 
    * To be fair, I was still playing with PANDAS in the initial 
    stage and hadn't understood its capacity.
  * Trust your tool.
  	* Wes McKinney does amazing work. Don't doubt his optimizations.
* Sometimes ad-hoc is best.
  * Doing some things, like visualizations helped me really understand 
  how to use a tool in practice, helping me expand it to more general 
  data sets later (I redid some methods to use in my other Kaggle challenges).

###Potential Improvements

* Anomaly detection to filter out bad examples.
* More extensive and data specific sampling for both visualizations and models.
* Using Principal Component Analysis to reduce dimensionality both for better visuals
and lower variance (at least for RandomForest)

