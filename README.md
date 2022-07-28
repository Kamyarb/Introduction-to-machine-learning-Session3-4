# Introduction-to-machine-learning-Session3-4
هدف: آشنایی با کتابخانه های Numpy & Pandas

مرور جبرخطی و ماتریس ها با استفاده از کتابخانه Numpy
حل مثال های عملی روی دیتاست های محصولات کشاورزی با استفاده از کتابخانه Pandas

هدف: یادگیری مصور سازی داده و آشنایی با کتابخانه های Matplotlib & Plotly

آشنایی با EDA و تحلیل اکتشافی
آشنایی با Matplotlib و رسم نمودار
آشنایی با Plotly و نمودار های تعاملی و پیشرفته
تحلیل اکتشافی دیتاست تایتانیک با مصور سازی
تحلیل اکتشافی دیتاست خانه های مسکونی با مصور سازی


# 🌟 NumPy -- Numerical Python 🌟  
  
<div>
<img src="https://i.postimg.cc/bvxpFgty/data.png" width="700"/>
</div>

One of the most important foundational package for numerical computing in Python.

* ndarray: multi-dimensional array
* mathematical functions
* linear algebra, random number generation, and so on  

NumPy based algorithes are generally 10 to 100 faster (or more) than pure Python algorithms

### Differences between lists and NumPy Arrays
* An array's size is immutable. You cannot append, insert or remove elements, like you can with a list.
* All of an array's elements must be of the same data type.
* A NumPy array behaves in a Pythonic fashion. You can len(my_array) just like you would assume.

>💥Numpy arrays are memory efficient.

### Numpy Array (ndarry) can have multi-dimension
![numpy_array](https://i.postimg.cc/sgSrvmKY/numpyarray2.png)


## Creating N-dimensional NumPy arrays
There are a number of ways to initialize new numpy arrays, for example from

* a Python list or tuples
* using functions that are dedicated to generating numpy arrays, such as `numpy.arange`, `numpy.linspace`, etc.
* reading data from files


From Lists:  
To create new vector and matrix arrays using Python lists we can use the `numpy.array()` function.


NumPy provides many functions for generating arrays. Some of them are:

* **numpy.arange()** - similar to `range()`function in python. we can create an array within a floating point step-wise range
* **numpy.linspace()** - create an evenly spaced sequence in a specified interval
* **numpy.random()** - create an array with random values
* **numpy.zeros()** - create an array of zeros
* **numpy.ones()** - create an array of ones
* **numpy.eye()** - create Identity Matrix 


### Random Number Generation
NumPy offers the random module to work with random numbers.


* **rand()**	Random values in a given shape.
* **randn()**	Return a sample (or samples) from the “standard normal” distribution.
* **randint()**	Return random integers from low (inclusive) to high (exclusive).
* **random()**	Return random floats in the half-open interval [0.0, 1.0).
* **seed()**  init random gen

### NumPy Array Attributes
The most important attributes of an ndarray object are:

<!-- * **ndarray.ndim** - the number of axes (dimensions) of the array. -->
* **ndarray.shape** - the dimensions of the array. For a matrix with n rows and m columns, shape will be (n,m).
* **ndarray.size** - the total number of elements of the array.
* **ndarray.dtype** - numpy.int32, numpy.int16, and numpy.float64 are some examples.


### NumPy Array Methods

Let's discuss some useful methods of numpy array:
* **reshape()** : Returns an array containing the same data with a new shape.  
These are useful methods for finding max, min or mean values. Or to find their index locations using argmin or argmax:
  * **min()**
  * **max()**
  * **argmax()**
  * **argmin()**
  * **mean()**
  
  
  ### more on reshape method in numpy  
numpy.reshape(a, newshape)  
* **a:** arraylike Array to be reshaped
* **newshape:** int or tuples of ints Should be compatible with the original shape. If an integer, then the result will be a 1-D array of that length. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.

![reshape](https://i.postimg.cc/g0dfrWxp/np-reshape.png)

## Accessing Arrays - Slicing and Indexing

* Numpy supports simple indexing (as in Python)
* Additionally, fancy indexing methods
  * Boolean indexing
  * List indexing
  
### Simple Indexing
As you expect it from Python:

* [idx]
* [begin:end:stepsize]
  * Default values
   * begin = 0
   * end = last element
   * stepsize = 1
   * colons are optional
* Negativ indizes are counted from the last element.
  *-i is the short form of n - i with n begin the number of elements in the array
  
  
  ###  Boolean Indexing
**Boolean indexing** allows you to select data subsets of an array that satisfy a given condition.


> **❗ NOTE:** `np.multiply` does an element-wise multiplication of two matrices, whereas `np.dot` is the dot product of two matrices.
<table><tr>
<td> <img src="https://i.postimg.cc/50F4SJSr/multiplication2.png" style="width: 500px;"/> </td>
<td> <img src="https://i.postimg.cc/66n67932/multiplication.png"  style="width: 500px;"/> </td>
</tr></table>


## What is Pandas

pandas is a Python package providing fast, ﬂexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python
![pandas](https://i.postimg.cc/J7vJ87Fs/pandas.jpg)


### Data Structures
pandas introduces two new data structures to Python - **Series** and **DataFrame**, both of which are built on top of NumPy (this means it's fast).


### Series
A Series is a one-dimensional array-like object containing a sequence of values (of similar types to NumPy types) and an associated array of data labels, called its index. By default, each item will receive an index label from 0 to N, where N is the length of the Series minus one.


### Dataframes
A DataFrame is a tablular data structure comprised of rows and columns, akin to a spreadsheet, database table, or R's data.frame object. You can also think of a DataFrame as a group of Series objects that share an index (the column names).

![dataframe](https://i.postimg.cc/6pLGCrpZ/pandas-dataframe.png)

### Reading Data
Much more often, you'll have a dataset you want to read into a DataFrame.
![df](https://i.postimg.cc/c439YLt3/excel.jpg)



### Index-based selection

When using `[]` like above, you can only select from one axis at once (rows or columns, not both). For more advanced indexing, you have some extra attributes:
    
* `loc`: selection by label
* `iloc`: selection by position

These methods index the different dimensions of the frame:

* `df.loc[row_indexer, column_indexer]`
* `df.iloc[row_indexer, column_indexer]`


## Inspecting a DataFrame  

pandas has a variety of functions for getting basic information about your DataFrame. Some of them are:

* **head()** shows the first five rows of the dataframe by default but you can specify the number of rows in the parenthesis
* **tail()** shows the bottom five rows by default
* **shape** tells us how many rows and columns exist in a dataframe
* **columns** Get the column names from our DataFrame.
* **dtypes** Check data types of each column
* **info()** Get summary information about our index, columns, and memory usage.
* **describe()** Get summary statistics about the columns in our data.
* **unique() / value_counts()** Get the distinct values in a DataFrame column
  > value_counts is similar to unique, but instead of returning an array of unique values, it returns a series with the frequency for each value
  
  
  #### More on `describe()` funciton
> By default, the describe method only includes numeric columns.  
Specifying `include='all'` will force pandas to generate summaries for all types of features in the dataframe. Some data types like string type don’t have any mean or standard deviation. In such cases, pandas will mark them as NaN



### Data Cleaning
 A lot of times, the CSV file you're given, you'll have a lot of missing values in the dataset, which you have to identify.
 We can detect missing values in the dataset (if any) by using the **isnull()** method together with the **sum()** method.
 
 If you do end up having missing values in your datasets, be sure to get familiar with these two functions.

* **dropna()** - This function allows you to drop all(or some) of the rows that have missing values.
* **fillna()** - This function allows you replace the rows that have missing values with the value that you pass in.

![cleaning](https://i.postimg.cc/0QgrznTD/cleaning.jpg)


## Descriptive statistics
* **sum()**
* **max()**
* **min()**
* **mean()**
* **count()** - Number of non-null observations



### Using the axis parameter
* axis = 0 ==> represents the function is applied column-wise
* axis = 1 ==> means that the function is applied row-wise on the DataFrame
![axis](https://i.postimg.cc/13fzMbPD/axes.png)


### Combining Dataframes
In many real-life situations, the data that we want to use comes in multiple files. We often have a need to combine these files into a single DataFrame to analyze the data.



The **concat()** function in pandas is used to append either columns or rows from one DataFrame to another.
![concat](https://i.postimg.cc/TPb3vmKw/concat.png)


### apply function

Pandas.apply allow the users to pass a function and apply it on every single value of the Pandas series. It uses the function along the series provided, column (axis=0) or row (axis=1).

# Exploratory Data Analysis (EDA)

EDA is one of the most important first steps when dealing with new data.It allows us to better understand the data and the tools we can use in future steps.
![EDA](https://i.postimg.cc/dQ6tFJV8/EDA.jpg)

## Setting up our DataFrame
we'll be exploring the IMDB Movies dataset from Kaggle!

### Closer Look at the Variables
Training set contains 1000 samples and 16 variables:

*    **Poster_Link** - Link of the poster that imdb using
*    **Series_Title** - Name of the movie
*    **Released_Year** - Year at which that movie released
*    **Certificate** - Certificate earned by that movie
*    **Runtime** - Total runtime of the movie
*    **Genre** - Genre of the movie
*    **IMDB_Rating** - Rating of the movie at IMDB site
*    **Overview** - mini story/ summary
*    **Meta_score** - Score earned by the movie
*    **Director** - Name of the Director
*    **Star1,Star2,Star3,Star4** - Name of the Stars
*    **Noofvotes** - Total number of votes
*    **Gross** - Money earned by that movie


## Step1: Understand the data
An introductory step is to look at the content of the data to get an idea of what you're going to be dealing with.
We can gain insight on the data with the following commands:

* **df.shape** (row, column)
* **df.columns** (column titles)
* **df.head** (top 5 results in table format)
* **df.nunique** (count of unique values for each variable/dimension)
* **df.isnull().sum()** (number of missing values in the data set) 
* **describe()** (description of the data in the DataFrame)



## Step 2: Data Cleaning
Data cleaning is used to not only increase the data integrity, but to also make it more usable and understandable to humans.

**Broad Overview of Cleaning Steps**

* **Handle empty values:** you can delete them, fill them with a value that makes sense
* **Check for data consistency:** case may be important for strings, formatting, etc
* **Handle outliers**
* **Remove duplicates**
* **Validate correctness of entries:** age columns shouldn't contain text, for instance

![cleaning](https://i.postimg.cc/5y2JyF28/cleaning-data.png)



<table><tr>
<td> <img src="https://i.postimg.cc/yd6tyCCH/fillna1.png" alt="Drawing" style="width: 390px;"/> </td>
<td> <img src="https://i.postimg.cc/SQ1FVRKt/fillna2.png" alt="Drawing" style="width: 390px;"/> </td>
</tr></table>

### Check for duplicate data
* The **duplicated()** method returns a Series with True and False values that describe which rows in the DataFrame are duplicated and not. 
* The **drop_duplicates()** method returns DataFrame with duplicate rows removed

## Data Visualization

### Plotly
Plotly's Python graphing library creates interactive, publication-ready graphs. Within Plotly, there is Plotly Express, which is a high level API designed to be as consistent and easy to learn as possible.
![plotly](https://i.postimg.cc/Dy4DgcH8/plotly.png)


<div>
 <img src = "https://i.postimg.cc/fLbq07YW/titanic.jpg" width="300 px" />
</div>

![newplot (7)](https://user-images.githubusercontent.com/100142624/181497548-5356f7d3-d1e2-4bc6-b30d-75651746db26.png)


![newplot (8)](https://user-images.githubusercontent.com/100142624/181497571-cf540be4-ac61-45b9-8e00-6e18d629c3c7.png)
![newplot (9)](https://user-images.githubusercontent.com/100142624/181497590-0449832e-7332-4efe-b7e4-0f8fd3917653.png)
![newplot (10)](https://user-images.githubusercontent.com/100142624/181497609-91cefbd2-bbd9-4ea2-8f32-4d067294a42f.png)
![newplot (11)](https://user-images.githubusercontent.com/100142624/181497625-eac287c2-254a-46b5-b7a1-f3cf6759952b.png)
![newplot (12)](https://user-images.githubusercontent.com/100142624/181497668-d18745a2-6ff3-43f9-ace3-8558ad68b6a5.png)


## Exploratory Data Analysis Tools (2022)
Imagine being able to generate insights from your data within minutes, whilst a data scientist takes over an hour using R or Python. That’s the benefit of using these exploratory data analysis tools.

The five most popular Python EDA tools: DataPrep, Pandas-profiling, SweetViz, Lux, and D-Tale. We will be focusing on D-Tale in this lecture.  
**DTale** is a Graphical Interface where we can select the data we want to analyze and how to analyze using different graphs and plots.   
![EDA](https://i.postimg.cc/SQYFTqzS/EDA2.png)

`pip install dtale`


![newplot (13)](https://user-images.githubusercontent.com/100142624/181498426-90735376-d819-4d1b-b0be-5895ad527e9c.png)

![newplot (14)](https://user-images.githubusercontent.com/100142624/181498479-f679895c-530a-4890-879e-00f38e96f981.png)


