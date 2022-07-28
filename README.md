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
