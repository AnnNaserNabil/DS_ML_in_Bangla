# পাইথনে ডাটা এক্সপ্লোরিং - ডাটা ভিজুয়ালাইজেশন


এই নোটবুকে আমরা কিছু স্ট্যাটিস্টিক্যাল এনালাইসিস ও তার ভিজুয়ালাইজেশন দেখব গ্রাফের মাধ্যমে। 

### ডাটা লোড করে নিই সবার আগে


শুরু করার আগে আমরা আমাদের আগের নোটবুকে ব্যবহার করা ছাত্রদের পড়ার সময় এর ডাটা লোড করে নিই। এটায় আগের বারের মত কারা কারা পাস করেছে তাদের হিসাব টা বের করে নিব। 


নিচের কোড গুলো রান করার মাধ্যমে ডাটা দেখতে পাব।




```python
import pandas as pd

# গিটহাব থেকে ডটাসেট লোড করে পান্ডাস এর ডাটাফ্রেমে লোড করে নিচ্ছি

df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')

# মিসিং ডাটা ট্রিম করে নিচ্ছি
df_students = df_students.dropna(axis=0, how='any')


#৬০ নাম্বারের বেশি মার্ক পাওয়া স্টুডেন্টদের আলাদা করে নিই
passes  = pd.Series(df_students['Grade'] >= 60)

# নতুন ডাটাকে একটা ডাটাফ্রেমে সেভ করছি
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)


# নতুন ডাটাফ্রেম
df_students 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>StudyHours</th>
      <th>Grade</th>
      <th>Pass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dan</td>
      <td>10.00</td>
      <td>50.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.50</td>
      <td>50.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro</td>
      <td>9.00</td>
      <td>47.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosie</td>
      <td>16.00</td>
      <td>97.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ethan</td>
      <td>9.25</td>
      <td>49.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vicky</td>
      <td>1.00</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frederic</td>
      <td>11.50</td>
      <td>53.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jimmie</td>
      <td>9.00</td>
      <td>42.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rhonda</td>
      <td>8.50</td>
      <td>26.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Giovanni</td>
      <td>14.50</td>
      <td>74.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Francesca</td>
      <td>15.50</td>
      <td>82.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rajab</td>
      <td>13.75</td>
      <td>62.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Naiyana</td>
      <td>9.00</td>
      <td>37.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kian</td>
      <td>8.00</td>
      <td>15.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Jenny</td>
      <td>15.50</td>
      <td>70.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Jakeem</td>
      <td>8.00</td>
      <td>27.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Helena</td>
      <td>9.00</td>
      <td>36.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ismat</td>
      <td>6.00</td>
      <td>35.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Anila</td>
      <td>10.00</td>
      <td>48.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Skye</td>
      <td>12.00</td>
      <td>52.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Daniel</td>
      <td>12.50</td>
      <td>63.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.00</td>
      <td>64.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## ম্যাটপ্লটলিবের মাধ্যমে ডাটা ভিজুয়ালাইজ করছি 


ডাটাফ্রেমের মাধ্যমে ডাটা বেশ ভালো ভাবে এক্সপ্লোর করা গেলেও কিছু কিছু সময় বিশাল ডাটাবেস নিয়ে কাজ করার সময় ডাটার স্ট্যাটিস্টিক্যাল ভিজুয়ালাইজেশন দেখে নিলে অনেক ভাল ভাবে কাজ করা যায়। এ ক্ষেত্রে ম্যাটপ্লটলিব বেশ সাহায্য করতে পারে। 

শুরুতে ছাত্ররা কারা কোন গ্রেড পেয়েছে তার একটা বার চার্ট দেখে নিই। 


```python
# নোটবুকের ইনলাইনে ডাটা ভিউজুয়ালাইজ করার জন্য ইনলাইন কমান্ড রান করে নিতে হবে
%matplotlib inline
from matplotlib import pyplot as plt

# ছাত্রদের নামের বিপরীতে প্রাপ্ত নাম্বারের বার চার্ট প্লট করে নিই
plt.figure(figsize=(30, 6)) 
plt.bar(x=df_students.Name, height=df_students.Grade)
        
# প্লট করা চার্ট দেখতে 
plt.show()
```


    
![png](/assets/SentimentAnalyis/output_30_0.png)
    


এইটা কাজ করেছে। আমরা প্লট থেকে একটা প্রাথমিক ধারনা পাচ্ছি। এই চার্টে আরো কিছু ইম্প্রুভমেন্ট করা যেতে পারে যাতে আমরা আরো ক্লিয়ার ভাবে আমাদের এনালাইসিস সম্পর্কে ধারনা পেতে পারি। 

এমরা ম্যাটপ্লটলিবের **pyplot** মডিউল ব্যাবহার করে চার্টটি তৈরি করতে পারি। এই মডিউলে বার চার্টের আরো কিছু ভিজুয়াল এলিমেন্ট যোগ করা যায়। যেমন,

-বার চার্টের জন্য স্পেসিফিক কিছু কালার এড করা যায়।

-  চার্টের একটা টাইটেল এড করা যায়। 

- এক্স ও ওয়াই অক্ষে লেবেলিং করা যায়

- বারর চার্টে গ্রিড এড করা যায় যাতে আরো ভালো ভাবে ভ্যালু বোঝা যায়

-  এক্স মার্কার গুলো রোটেট করা যাবে। 



```python
# ছাত্রদের নাম আর গ্রেডের একটা বার চার্ট তৈরি করে নিই   
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# চার্টকে কাস্টমাইজ করি 
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# চার্ট শো করি
plt.show()
```


    
![png](/assets/SentimentAnalyis/output_5_0.png)
    


প্লট **Figure** এর ক্ষেত্রে আগের উদাহরনে চার্ট আমাকে তৈরি করে দিলেও এখানে নিজের মত করে চার্ট কাস্টমাইজ করে নিতে পারছি। যেমন নিচের কোড দিয়ে একটা স্পেসিফিক সাইজের প্লট তৈরি করে নিতে পারব আমরা। 


```python
# প্লটের ফিগার সাইজ ঠিক 
fig = plt.figure(figsize=(8,3))

# প্লট তৈরি করলাম   
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# প্লট কাস্টমাইজ করে 
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# ফিগার শো করার জন্য
plt.show()
```


    
![png](/assets/SentimentAnalyis/output_7_0.png)
    


একটা ফিগারে বেশ কয়েকটা সাবপ্লট নেয়া যায়। 


যেমন নিচের ছবিতে দুইটা সাব প্লট থাকবে, একটায় স্টুডেন্ট গ্রেডের বার চার্ট আরেকটায়  একটা পাই চার্ট, পাস করা আর ফেইল করা ছাত্রদের নাম্বার কম্পেয়ার করে এমন। 


```python
# দুইটা সাব প্লটের একটা ফিগ তৈরি করি (১ রো , ২ কলাম ) 
fig, ax = plt.subplots(1, 2, figsize = (10,4))

# নাম্বার আর গ্রেডের বার প্লট তৈরি চার্ট তৈরি করে চার্টনিই
ax[0].bar(x=df_students.Name, height=df_students.Grade, color='orange')
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students.Name, rotation=90)

# পাশে আরেকটা পাই চার্ট
pass_counts = df_students['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

# ফিগারের নাম দিচ্ছি 
fig.suptitle('Student Data')

# দেখে নিই
fig.show()
```

    /tmp/ipykernel_15071/3078456740.py:7: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      ax[0].set_xticklabels(df_students.Name, rotation=90)
    /tmp/ipykernel_15071/3078456740.py:19: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_9_1.png)
    


Until now, you've used methods of the `Matplotlib.pyplot` object to plot charts.
এতক্ষন পর্যন্ত পান্ডাস এর পাই প্লট ব্যাবহার করে চার্ট তৈরি করছিলাম। পাইথনের বেশ কিছু প্যাকেজে ম্যাটপ্লটলিব ইন্ট্রিগ্রেটেড আছে যাতে করে সেই মডিউলের মধ্যেই ম্যাটপ্লটলিব ব্যাবহার করে চার্ট তৈরি করা যায়। যেমন পান্ডাস ডাটাফ্রেমেই প্লট করার একটা টুল আছে, সেটা দিয়েও প্লট করা যায়।


```python
df_students.plot.bar(x='Name', y='StudyHours', color='teal', figsize=(6,4))
```




    <Axes: xlabel='Name'>




    
![png](/assets/SentimentAnalyis/output_11_1.png)
    


## স্ট্যাটিস্টিক্যাল এনালাইসিস শুরু করি 
এখন পর্যন্ত পাইথন দিয়ে ডাটা ম্যানুপুলেট ও ভিজুয়ালাইজ করলাম আমরা, এখন এনালাইসিস করব। 
ডাটাসায়েন্সের ভিত্তির অনেকটাই স্ট্যাটিস্টিক্স কেন্দ্রিক। আমরা স্ট্যাটিস্টিকস এর বেশ কিছু টেকনিক দেখব। 



### ডেস্ক্রিপটিভ স্ট্যাটিস্টিকস ও ডাটা ডিস্ট্রিবিউশ

আমরা যখন ভ্যারিয়েবল নিয়ে কাজ করি তখন সেই ভ্যারিয়েবল গুলোর ডিস্ট্রিবিউশন সম্পর্কে জানার চেষ্টা করি। পুরো ডাটা সেটে বিভিন্ন ভ্যালু কিভাবে ছড়িয়ে ছিটিয়ে আছে সেটা বোঝার চেষ্টা করি । হিস্টোগ্রাম ব্যাবহার করে একেকটা ভ্যারিয়েবলের অকারেন্স ফ্রিকোয়েন্সি দেখতে পাই। 








```python
# ভ্যারিয়েবল সিলেক্ট করি কলাম থেকে
var_data = df_students['Grade']

# একটা ফিগার নেই
fig = plt.figure(figsize=(10,4))

# ফিগারে হিস্টোগ্রাম প্লট করি
plt.hist(var_data)

# টাইটেল ও লেবেল দেই
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# ফিগার দেখি
fig.show()
```

    /tmp/ipykernel_15071/1979159869.py:16: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_13_1.png)
    



হিস্টোগ্রাম একটা সিমেট্রিক শেইপ। যেটায় যে ভ্যারিয়েবল গুলো বেশি সংখ্যক বার এসেছে মাঝে থাকে আর যেগুলো কম সংখ্যকবার এসেছে দুই পাশে থাকে
#### সেন্ট্রাল টেন্ডেন্সি  দেখি

ডিস্ট্রিবিউশনকে ভালো করে বোঝার জন্য আমরা সেন্ট্রাল টেন্ডেন্সি মেজার করতে পারি। এইটার মাধ্যমে ডাটা মিডল ভ্যালু গুলো বের করতে পারি। এর মাধ্যমে আমরা যে মিডল ভ্যালু গুলো বের করতে পারি 


-  *mean*: স্যাম্পল এভারেজ। টোটাল ডাটা / টোটাল নাম্বার 
-  *median*: স্যাম্পল ভ্যালুর মিডল রেঞ্জ 
-  *mode*: যে ভ্যালু সবচে বেশিবার এসেছে


এই ভ্যালু গুলো ক্যালকুলেট করি। এর সাথে ডাটাড় ম্যাক্সিমাম ও মিনিমাম ভ্যালু ক্যলকুলেট করি যাতে করে একসাথে কম্পেয়ার করা যায়। এদের একটা হিস্টোগ্রামে এড করে নিই। 



```python
# ভ্যারিয়েবল সিলেক্ট করি
var = df_students['Grade']

# স্ট্যাটিস্টিক্স গুলো বের করি
min_val = var.min()
max_val = var.max()
mean_val = var.mean()
med_val = var.median()
mod_val = var.mode()[0]

print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                        mean_val,
                                                                                        med_val,
                                                                                        mod_val,
                                                                                        max_val))

# ফিগার তৈরি করি
fig = plt.figure(figsize=(10,4))

# হিস্টোগ্রাম প্লট করি
plt.hist(var)

# স্ট্যাটিস্টিক্স ভ্যালু গুলোর জন্য ভ্যালু লাইন এড করি
plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

# টাইটেল ও লেবেল এড করি
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# ফিগার শো করি
fig.show()
```

    Minimum:3.00
    Mean:49.18
    Median:49.50
    Mode:50.00
    Maximum:97.00
    


    /tmp/ipykernel_15071/1620963976.py:36: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_15_2.png)
    


গ্রেড ডাটায় মিন মিডিয়ান মোড সব গুলো প্রায় কাছাকাছি অবস্থান করছে, ৫০ এর আশেপাশেই। 


ডিস্ট্রিবিউশন দেখার আরেকটা ওয়ে হলো বক্স প্লট। একটা বক্স প্লট বানাই


```python
# ভ্যারিয়েবল নিচ্ছি
var = df_students['Grade']

# ফিগার তৈরি করছি
fig = plt.figure(figsize=(10,4))

# হিস্টোগ্রাম বানাচ্ছি
plt.boxplot(var)

# টাইটেল ও লেবেল এড করালাম 
plt.title('Data Distribution')

# ফিগার শো করি
fig.show()
```

    /tmp/ipykernel_15071/1836236302.py:14: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_17_1.png)
    


একটা বক্সপ্লট হিস্টোগ্রামের চেয়ে ভিন্ন ভাবে ডাটা শো করে। বক্স প্লট ডাটার দুই কোয়ার্টাইল কোথায় আছে সেটা দেখায়। এক্ষত্রে প্রায় অর্ধেক গ্রেড আছে ৩৬ আর ৬৩ এর মাঝে। বাকি অর্ধেক গ্রেড আছে ০-৩৬ আর ৬৩-১০০ এর মধ্যে। বক্সের মধ্যে যে লাইন টা থাকে ওটা মিডিয়ান লাইন। 

শেখার জন্য, হিস্টোগ্রাম আর বক্সপ্লটকে এক সাথে দেখলে ভালো করে ডাটা সম্পর্কে আইডিয়া নেয়া যায়। ক্ষেত্রে বক্স প্লটের এক্সিস রোটেট করে প্লট করলে একই ভ্যারিয়েবল লাইনে বক্সপ্লট ও হিস্টোগ্রাম থাকবে। 


```python
# রিইউজেবল একটা ফাংশন তৈরি করি দুইটা প্লটের জন্য 
def show_distribution(var_data):
    from matplotlib import pyplot as plt

    # স্ট্যাটিস্টিক্যাল হিসাব নিচ্ছি
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))

    # ২ টা সাবপ্লটের ফিগার তৈরি করে নিচ্ছি
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # হিস্টোগ্রাম প্লট করি 
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # হিস্টোগ্রামে মিন মিডিয়ান মোড লাইন এড করি 
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # বক্সপ্লট এড করি    
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # ফিগারে টাইটেল এড করি 
    fig.suptitle('Data Distribution')

    # ফিগার শো করি 
    fig.show()

# ভ্যারিয়েবল এড করি 
col = df_students['Grade']
# ফাংশন কল করি 
show_distribution(col)
```

    Minimum:3.00
    Mean:49.18
    Median:49.50
    Mode:50.00
    Maximum:97.00
    


    /tmp/ipykernel_15071/3002759621.py:40: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_19_2.png)
    


সেন্ট্রাল টেন্ডেন্সি মেজারে ক্ষেত্রে ভ্যালু গুলো ডাটা ডিস্ট্রিবিউশনের মাঝে আসতেসে, একটা সিমেট্রিক ভ্যালুর মাঝে শুরু হয়ে আস্তে আস্তে দুইপাশে কমে যায়। 

স্ট্যাটিস্টিক্সে আমরা স্যাম্পল ডাটা থেকে ভ্যালু নিয়ে পপুলেশন ডাটায় প্রেডিক্ট করতে পারি। যেমন এখানে ২২ জনের ডাটা নিয়ে আমরা আরো যত ছাত্রের গ্রেড ডাটা কালেক্ট করতে পারি তাদের সম্পর্কে একটা ধারনা করতে পারব। আমরা সাধারনত পপুলেশনের বিভিন্ন স্ট্যাটিস্টিক্স সম্পর্কে জানতে চাই কিন্তু এত বড় সাইজের ডাটা সংগ্রহ করা সময় সাপেক্ষ। এজন্য আমরা স্যাম্পল ডাটা নিয়ে পপুলেশন ডাটা প্রেডিক্ট করে থাকি। 
 
আমাদের কাছে যথেষ্ট ডাটা থাকলে আমরা ডাটার প্রবাবিলিটি ডেনসিটি ফাংশন সম্পর্কে জানতে পারি। যা ফুল পপুলেশনের ডিস্ট্রিবিশন এস্টিমেট করতে পারে। 

ম্যাটপ্লটলিবের **pyplot** ক্লাসে মাধ্যমে আমরা সেটা দেখতে পাই। 


```python
def show_density(var_data):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10,4))

    # ডেনসিটি প্লট করি
    var_data.plot.density()

    # টাইটেল ও লেবেল এড করি 
    plt.title('Data Density')

    # মিন মিডিয়ান মোড শো করি 
    plt.axvline(x=var_data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    # ফিগার শো করি
    plt.show()

# গ্রেডের ডেনসিটি বের করি
col = df_students['Grade']
show_density(col)
```


    
![png](/assets/SentimentAnalyis/output_21_0.png)
    


হিস্টোগ্রাম দেখেই বোঝা যাচ্ছিলো যে এই ডাটা টা বেল কার্ভের নরমাল ডিস্ট্রিবিউশন যার মিন আর মোড সেন্টারে আর দুই পাশে সিমেট্রিক টেইল আছে। 

## সারসংক্ষেপ

এখন পর্যন্ত আমরা যা দেখলাম 



১. ম্যাটপ্লটলিব দিয়ে গ্রাফ বানিয়েছি
২. গ্রাফগুলোকে কাস্টমাইজ করা শিখেছি
৩. বেসিক স্ট্যাটিস্টিক্সের অংশ হিসেবে মিন মিডিয়ান মোড বের করেছি
৪. হিস্টোগ্রাম আর বক্সপ্লট ব্যাবহার করে ডাটার স্প্রেড দেখেছি
৫. স্যাম্পল গ্রেড থেকে পপুলেশন গ্রেড কেমন হতে পারে বুঝতে চেয়েছি। 


পরের নোটবুকে আমরা আন ইউজুয়াল ডাটা বের  এবং ডাটা মধ্যকার সম্পর্ক দেখার চেষ্টা করব। 

## আরো পড়তে চাইলে

মডিউল গুলো সম্পর্কে আরো জানতে চাইলে

- [NumPy](https://numpy.org/doc/stable/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib](https://matplotlib.org/contents.html)

# ধন্যবাদ


```python

```
