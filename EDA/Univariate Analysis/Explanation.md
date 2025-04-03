# Univariate Analysis

While doing EDA process the first step we do is to perform *Univariate Analysis*

Generally the Dataset is divided into two types of columns
<ul>
<li>
Categorical
</li>
<li>
Numerical
</li>
</ul>

so there are different ways to analyse both columns 

## Categorical

In categorical to perform Univariate Analysis we can plot 

<ul><li><b>Countplot</b></li></ul>

![image](https://github.com/user-attachments/assets/b75d2f11-d6c5-413b-b759-fd2ae9e1861d)


seaborn code snippet - `sns.countplot(df[''])`

pandas code snippet - `df[''].value_counts().plot(kind='bar')`

<ul><li><b>Pie Chart</b></li></ul>

code with %age - `df[''].value_counts().plot(kind='pie', autopct = %.2f%)`

code without %age - `df[''].value_counts().plot(kind='pie')`

## Numerical

In numerical columns to perform Univariate Analysis we plot

<ul><li><b>Histogram</b></li></ul>

seaborn code snippet - `sns.hist(df[''])`

pandas code snippet - `df[''].value_counts().plot(kind='bar')`




