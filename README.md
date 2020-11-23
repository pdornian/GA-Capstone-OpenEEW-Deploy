The following is the final project I completed for General Assembly's Data Science Intensive program. 

**The remainder of the readme below is left unchanged from our project's submission.**

# Predicting Earthquake Location From Accelerometer Data

## Introduction

### Background

The west coast of Mexico is a hotbed of seismic activity and an ideal testing ground for new seismic detection technology. Since 2017, [Grillo(tm)](https://grillo.io/) has been developing and maintaining an IoT network of accelerometer based seismic sensors in the region with a focus on early earthquake warning (EEW) alerts. Their devices have performed well and generated comparable results as a warning system to national scientfic institutes using laboratory grade sesmic tools. 

Grillo has shared their data and technology via the [OpenEEW](https://openeew.com/) project. This is an open source initiative that allows any interested party to have access to years worth of their device readings through a freely available [AWS portal](https://s3.amazonaws.com/grillo-openeew/index.html).

The Mexican Servicio Sismológico Nacional provides a [comprehensive searchable database](http://www2.ssn.unam.mx:8080/catalogo/) of detected seismic activity in the country with logs of location, time, and earthquake magnitude.

Cross referencing the above two data sources allows one to consider a variety of interesting questions.

### Problem Statement
Given the five minutes of data immediately after an earthquake start from each of the 27 actively maintained OpenEEW devices, can we identify the epicentre of the earthquake to some degree of accuracy?

Earthquake epicentres are typically located via triangulation by comparing readings from three or more seismographs. A brief investigation of literature suggests that accelerometers are typically considered to be poor devices for determining distances and locations due to not measuring displacment and being highly influenced by local geological conditions. However, due to the close proximity of the OpenEEW sensors and the large amount of data concentrated in small area in southwest Mexico, it seems plausible that we might be able to develop some predictive ability at a local level.

Disclaimer: I am not a geologist, a physicist, or a geophysicist.

### Answer: TL:DR Version
The conventional wisdom from geophysicists seems right. A variety of convolutional neural networks were trained in an attempt to make headway into the problem, but predictions with a statistically signficant degree of accuracy were not achieved.

An interactive map demonstrating predictions from one such model is available at: <https://ga-openeew.herokuapp.com/> .

A detailed description of methodology and attemps follows below.

## Project Description

Some of the data and model files are not included in this repository due to GitHub size limits. This will prevent some of the notebook code from running.

Missing data includes:
- top_1000_final.p : Accelerometer data of the top 1000 earthquakes in magnitude.
- epicentre_prediction (folder): Keras model used to generate earthquake location predictions.

### Data Gathering, EDA and Preprocessing

#### Earthquakes

The most complete records from OpenEEW are available from 2018 onwards, so I decided to analyze earthquakes from 2018 to 2020. Due to large filesize and runtime issues, the decision was made to focus on the 1000 strongest seismic events recorded over those two years (9000 events of magnitude > 4 were recorded).

Earthquakes were predominantly recorded off the southwest coast of Mexico. The next most seismically active area was the Baja California region, but I ultimately removed these earthquakes from the dataset due to their distance from the seismic devices, leaving us with a total set of 928 earthquake events.

![Outlier map coloured](/images/outliers.PNG)

#### Accelerometer Readings

An online device uploads its readings every five minutes to the OpenEEW AWS. 28 devices were identified as being regularly active from 2018 to 2020, although each had significant stretches of downtime and/or lost readings. 

![acc readings](/images/accelerometers.PNG)

Each device records an acceleration measure in gals (1 cm/s^2) on the x, y, and z axis at a rate of 32 times per second. To conserve space and summarize, retrieved data was averaged to record a single reading per second for each axis.

In an abuse of terminology, the primary unit of analysis is what we will call a **waveform**: a 3 x 300 array of data from a given device recording its average per-second reading on each axis for five minutes. The average earthquake lasts ten to thirty seconds. Five minutes of data is likely excessive, but was chosen to account for the time it takes seismic waves to travel to ensure that a given earthquake may be represented in all active devices (if high enough magnitude to be detectable).

For each of the 928 seismic events, waveforms were retrieved for each device starting at one second before the earthquake start by querying and processing data from the OpenEEW AWS.  

Missing data was naively interpolated as the zero matrix. 

### Modeling

The majority of modelling experimentation was done in Keras using convolutional neural network methods, attempting to predict a (latitude, longitude) tuple. Data was split 80-20 into a training and test set. 

![network structure](/images/nn_structure.PNG)

A convolutional layer with a 3x1 kernel was first applied to try to identify correlated waveform features across the x, y, and z axis and then max pooled. The resulting 1x300 vector was then passed through another convolutional layer with a 1x2 kernel to identify features shared across the time axis. This was then passed through a dense feed forward neural network. 

Numbers of nodes, hidden layers, and batch sizes were varied. A variety of regularization methods were applied. Learning rates were adjusted both statically and dynamically. However, no statistically significant results were achieved. Latitude and longitude almost inevitable converged to the mean or were returned with unpredictable results. Convergence to the mean was even more pronounced in neural networks that predicted just one of the location attributes. 

For demonstration and exploration purposes, a deliberately overfit model is implemented in the following Heroku app.
https://ga-openeew.herokuapp.com/

The model's predictions on training and test data may be seen below. Actual earthquake locations are marked in red, and locations predicted based on waveform data are marked in purple.

Training Set Predictions:
![network structure](/images/train_preds.png)

Test Set Predictions:
![network structure](/images/test_preds.png)

Though visually plausible, analysis of R^2 scores immediately demonstrates massive overfitting and poor predictive ability.

Train R^2: 0.780

Test R^2: -0.355

-----------

Many additional (but less thoroughly documented) attempts were made to find some semblence of predictive ability in the accellerometer data, with similar inconclusive results. 

I experimented with reducing the waveforms to the peak 30 seconds of activity in each 5 minute sample to minimize noise and low activity data. When this failed to produce results, I tried simplifying further by creating a "max impulse score", summing activity in each waveform to a single scalar value. With data now representable in vector form, I opened up the sklearn toolkit and attempted predictions using random forest methods (among others). However, I wasn't able to find anything that could produce a model with non-negative cross validation scores.

Attempts were made to convert the problem into a classification model where given waveform data, a model would try to guess the nearest Mexican state to the earthquake epicentre. All trained models performed at baseline level at best.

Some attempts were made to try to predict earthquake magnitude given waveform data. These were similarly inconclusive. 

## Conclusions

Predicting earthquake epicentre location from accelerometer data is a non-trivial problem, and probably requires more geological expertise than the author currently has. No models with demonstratably significant predictive ability were found. 

### Likely Errors/Confounding Factors

**Lack of Magnitude/Significance**: Of the 928 earthquakes analyzed, only 27 were classified as "strong" (magnitude >= 5.5) by the SSN. The vast majority of data corresponded to earthquakes of magnitude near 4, and most of the earthquakes had epicentres located off shore. Such earthquakes are barely felt by humans, and likely produced negligable accelerometer readings. I should have done more thorough EDA to identify what kind of readings were generated by small seismic events to see if they were measurable in the given dataset or not.

**Missing Data/Invalid Interpolation**: I was hesitant to drop missing data -- there wasn't a single earthquake that all 27 devices were active for, but I wanted to make use of all the devices due to them each being useful for certain time ranges. Interpolating missing data as 0 matrices was likely a poor decision, as it carries a strong semantic meaning of "no seismic activity". Absence of data should not be confused with low seismic readings. 

### Opportunities for Revision/Expansion

The OpenEEW dataset is extremely robust and has great possibilities for further exploration. My approaches were mostly naive in the time dimension, and there is high potential for time series analysis using the dataset. Despite the name "Early Earthquake Warning" with the whole purpose of the project being earthquake detection in as quick a time as possible, my analysis was relatively time-agnostic, which probably wasn't a great use of the data.

Despite the lack of results for this given problem, using convolutional neural networks to analyze waveform shape seems like a plausible method of prediction. I would be interested in swapping out accelerometer readings for a dataset from precise seismographs to investigate the methodology further.
