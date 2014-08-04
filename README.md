#Allstate Prediction Challenge

##Exploratory Analysis

###Feature Change Analysis

Most of the visualization and explorative analysis is generated in the [visuals.py] 
script and in IPython. The script functions as a summary of the important 
parts of my IPython explorations. The first analysis I did was look at the 
changes in the customers and the policies they view in [feature_changes.png]. 
In this analysis I notice a couple things:
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

Now, let's look at the policy changes more closely, specifically between 
the first and last shopping points, as in [feature_changes.png] and the 
second to last shopping point and the last shopping point, as in 
[Second_to_last_polcy_changes.png] and [Second_to_last_polcy_changes_by_Degree.png]

The probability of a customer purchasing his/her last view insurance policy scheme
is 64.33%. This was calculated by taking the areas of the tails of the histogram,
excluding the rectangle above 0.

##Challenges

* Clearly, from [missings.png], something needs to be done about the 
risk_factor column if it's to be used for in the prediction model.
* Can the second to last viewed policy be utiilized to help predict 
the policy the customer actually purchases?

##Model Analysis



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

