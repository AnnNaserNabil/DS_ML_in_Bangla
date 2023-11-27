# নামপাই ও পান্ডাস

## নামপাই দিয়ে ডাটা এক্সপ্লোর করি

পাইথনে ডাটা নিয়ে কাজ করা যায় কিন্তু শুধু পাইথনে লিস্ট আকারে ডাটা স্টোর করা যায়। নিচে একটা ডেমো ডাটাবেজ বানিয়ে দেখাচ্ছি।


```python
data = [50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
print(data)
```

    [50, 50, 47, 97, 49, 3, 53, 42, 26, 74, 82, 62, 37, 15, 70, 27, 36, 35, 48, 52, 63, 64]


এই ডাটা টা পাইথনের **list** স্ট্রাকচারে বানানো, যা ডাটা ম্যানিপুলেশন করার জন্য বেশ ভালো হলেও ডাটা এনালাইসিস করার জন্য সুবিধা জনক না। এজন্য আমরা একটা পাইথন প্যাকেজ ব্যাবহার করব, সেটা হলো পাইথনের নামপাই(NumPy) প্যাকেজ/লাইব্রেরী। এটায় পাইথনের সাহায্যে ডাটা এনালাইসিস করার জন্য বিভিন্ন ডাটা টাইপ নিয়ে কাজ করা যায়। 

নিচে **array** স্ট্রাকচারে ডাটা গুলো লোড করা হচ্ছে।


```python
import numpy as np

grades = np.array(data)
print(grades)
```

    [50 50 47 97 49  3 53 42 26 74 82 62 37 15 70 27 36 35 48 52 63 64]


লিস্ট স্ট্রাকচার আর নামপাই এর এরের মধ্যে পার্থক্য কি ? দুইটা ডাটাস্ট্রাকচারকে ২ দিয়ে গুন করে দেখে নিচ্ছি তাহলে। 



```python
print (type(data),'x 2:', data * 2)
print('---')
print (type(grades),'x 2:', grades * 2)
```

    <class 'list'> x 2: [50, 50, 47, 97, 49, 3, 53, 42, 26, 74, 82, 62, 37, 15, 70, 27, 36, 35, 48, 52, 63, 64, 50, 50, 47, 97, 49, 3, 53, 42, 26, 74, 82, 62, 37, 15, 70, 27, 36, 35, 48, 52, 63, 64]
    ---
    <class 'numpy.ndarray'> x 2: [100 100  94 194  98   6 106  84  52 148 164 124  74  30 140  54  72  70
      96 104 126 128]


এখানে লিস্টকে ২ দিয়ে গুন করলে লিস্ট টা পর পর দুই বার প্রিন্ট হয়েছে। আর অন্য দিকে দেখা যাচ্ছে নামপাই এর এরেতে ভেক্টোরের মত নাম্বার গুন হয়েছে। নামপাই এর এরে ম্যাথমেটিক্যাল ক্যালকুলেশন করার জন্য বিশেষ ভাবে তৈরি করা হয়েছে। যা সাধরন লিস্ট স্ট্রাকচার দিয়ে করা সহজ না। 

নামপাই n সংখ্যক ডাইমেনশন নিয়ে কাজ করতে পারে। নামপাই তে  **numpy.ndarray**. The **nd** তে n হলো ডাইমেনশনের সংখ্যা। নামপাই এর .shape মডিউল দিয়ে ডাটার ডাইমেনশন দেখা যায়। নিচে একবার আমাদের তৈরি করা ডাটা স্ট্রাকচারে ডাইমেনশন দেখে নিচ্ছি। 



```python
grades.shape
```




    (22,)



.shape মডিউলের মাধ্যমে আমরা দেখতে পাচ্ছি যে ডাটা স্ট্রাকচারে ২২ টা এলিমেন্ট আছে। ইনডেক্সিং এর মাধ্যমে ডাটা স্ট্রাকচারে প্রথম এলিমেন্ট টার পজিশন সব সময় ০ থেকে শুরু হয়। ০,৫,৬ এমন পজিশন ব্যাবহার করে কোরেস্পন্ডিং পজিশনে থাকা ডাটা দেখতে পাব আমরা। 



```python
grades[0]
```




    50




```python
grades[20]
```




    63



নামপাই দিয়ে বেসিক কিছু ডাটা এনালাইসিস করে ফেলি তাহলে এখন। শুরুতেই ২২ টা ডাটার এভারেজ/মিন বের করি। এজন্য নামপাই এর .mean() মডিউল ব্যাবহার করে নিচ্ছি। 


```python
grades.mean()
```




    49.18181818181818



স্টুডেন্ট দের মিন গ্রেড হলো ৫০ এর কাছাকাছি। ০-১০০ এর মাঝামাঝিতে। 

এখন ডাটা স্ট্রাকচারে ঐ স্টুডেন্ট গুলোর জন্য আরেক সেট ডাটা এড করছি। এবার প্রতি সপ্তাহে কত ঘন্টা সময় তারা পড়া শোনায় ব্যায় করে সেটা এড করব। 


```python
# স্টাডি আওয়ারের একটা এরেকে ডিফাইন করছি শুরুতে
study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,
               13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]

# আগের এরে আর নতুন এরে যোগ করে আরেক্টা নতুন টু ডাইমেনশনাল এরে বানাচ্ছি 
student_data = np.array([study_hours, grades])

# নতুন বানানো এরে দেখে নিই
student_data
```




    array([[10.  , 11.5 ,  9.  , 16.  ,  9.25,  1.  , 11.5 ,  9.  ,  8.5 ,
            14.5 , 15.5 , 13.75,  9.  ,  8.  , 15.5 ,  8.  ,  9.  ,  6.  ,
            10.  , 12.  , 12.5 , 12.  ],
           [50.  , 50.  , 47.  , 97.  , 49.  ,  3.  , 53.  , 42.  , 26.  ,
            74.  , 82.  , 62.  , 37.  , 15.  , 70.  , 27.  , 36.  , 35.  ,
            48.  , 52.  , 63.  , 64.  ]])



নতুন ডাটা স্ট্রাকচারটা টু ডাইমেনশনাল এরে। দুইটা এরের এরে হলো এই নতুন এরে টা। এর শেইপ দেখে নিচ্ছি আবার। 



```python
# এরের শেইপ দেখে নিই
student_data.shape
```




    (2, 22)



এখানের **student_data** এরের দুইটা এলিমেন্ট। প্রত্যেকটাই একেকটা এরে যাদের প্রত্যেকটার এলিমেন্ট সংখ্যা হলো ২২। '


এই ডাটা স্ট্র্যাকচারের কোথায় কি আছে দেখার জন্য প্রত্যেকটা ডাটা স্পেসেফিক পজিশুন জানতে হবে। যেমন প্রথম এরের প্রথম ভ্যালু জানতে চাইলে (প্রথম স্টুডেন্টেড় স্টাডি আওয়ার দেখা জন্য)


```python
# প্রথম এলিমেন্টের প্রথম এলিমেন্ট জন্য ।
# প্রথম এরের প্রথম এলিমেন্ট দেখার জন্য । 
student_data[0][0]
```




    10.0




এখন আমাদের কাছে স্টুডেন্ট গুলোর গ্রেডের পাশাপাশি তাদের সাপ্তাহিক পড়ার সময় আছে। এটার মাধ্যমে স্টুডেন্ট গ্রেডের সাথে পড়ার সময় কম্পেয়ার করতে পারব আমরা।  


```python
# দুইটা সাব এরের এভারেজ বের করে নিই আগে। এখানে ডাটা ফ্রেমের শেপের প্রথম ডিজিট হলো সাব এরের আইডেন্টিফাইয়ার। 
avg_study = student_data[0].mean()
avg_grade = student_data[1].mean()

print('Average study hours: {:.2f}\nAverage grade: {:.2f}'.format(avg_study, avg_grade))
```

    Average study hours: 10.52
    Average grade: 49.18


# পান্ডাস দিয়ে টেবুলার ডাটা এক্সপ্লোর করি


নামপাই দিয়ে দুই ডাইমেনশনাল ডাটা স্ট্রাকচার দেখা গেলেও পান্ডাস দিয়ে আরো ভাল করে ডাটা এক্সপ্লোর করা যায়। পান্ডাস এর ডাটা ফ্রেমে ডাটা এক্সপ্লোর করার জন্য ব্যাবহার করা ভালো।  এখানে আমরা একটা ডেমো ডাটা ফ্রেম তৈরি করে নিচ্ছি যার প্রথম কলামে নাম থাকবে আর এর সাথে বাকি দুই কলাম আগের বানানো নামপাই এর ডাটা ফ্রেম থেকে নিয়ে নিচ্ছি । প্রথম কলাম নাম, সেকেন্ড কলামে স্টুডেন্ট ডাটার স্টাডি হাওয়ার আর পরের থার্ড কলামে স্টুডেন্ট ডাটা থেকে গ্রেড নিয়ে নিচ্ছি। 


```python
import pandas as pd

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 
                                     'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],
                            'StudyHours':student_data[0],
                            'Grade':student_data[1]})

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dan</td>
      <td>10.00</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.50</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro</td>
      <td>9.00</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosie</td>
      <td>16.00</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ethan</td>
      <td>9.25</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vicky</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frederic</td>
      <td>11.50</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jimmie</td>
      <td>9.00</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rhonda</td>
      <td>8.50</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Giovanni</td>
      <td>14.50</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Francesca</td>
      <td>15.50</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rajab</td>
      <td>13.75</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Naiyana</td>
      <td>9.00</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kian</td>
      <td>8.00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Jenny</td>
      <td>15.50</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Jakeem</td>
      <td>8.00</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Helena</td>
      <td>9.00</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ismat</td>
      <td>6.00</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Anila</td>
      <td>10.00</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Skye</td>
      <td>12.00</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Daniel</td>
      <td>12.50</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.00</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>



ডাটা ফ্রেমে অলরেডি একটা ইনডেক্স ভ্যালু বসেছে প্রত্যেকটা রোতে। এটাতে অন্য কোনো কিছু দিয়েও ইনডেক্সিং করা যেত। যেমন ইমেইল এড্রেস দিয়ে ইনডেক্সিং করা যায়। এখানে বাই ডিফল্ট একটা ইনডেক্সিং হয়েছে যাতে করে আমরা সহজেই প্রত্যেকটা রো কে আইডেন্টিফাই করতে পারি। 

###  DataFrame থেকে ডাটা বের করি


***loc*** মেথড ব্যবহার করে ডাটা ফ্রেম থেকে স্পেস্ফিক ডাটা ভ্যালু বের করা যায়।  নিচে ৫ নাম্বার ইন্ডেক্সিং এর রো এর ভ্যালু বের করে দেখে নিচ্ছি। 



```python
# ইনডেক্স ভ্যালু ৫ এর জন্য ডাটা বের করে নিচ্ছি
df_students.loc[5]
```




    Name          Vicky
    StudyHours      1.0
    Grade           3.0
    Name: 5, dtype: object



ইনডেক্সের একটা নির্দিষ্ট রেঞ্জের ডাটাও বের করে নিতে পারব আমরা। এখানে আমরা ০ থেকে ৫ পর্যন্ত ডাটা বের করে নিচ্ছি আমরা। 


```python
# ০ থেকে ৫ পর্যিন্ত ইনডেক্সের রো ডাটা বের করে নিচ্ছি। 
df_students.loc[0:5]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dan</td>
      <td>10.00</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.50</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro</td>
      <td>9.00</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosie</td>
      <td>16.00</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ethan</td>
      <td>9.25</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vicky</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



 **loc** মেথড ছাড়াও **iloc** মেথড ব্যাবহার করে রো এর অরিজিনাল পজিশনের জন্য ভ্যালু বের করা যায় । যেকোনো রকম ইনডেক্সের জন্যই।


```python
# প্রথম ৫ রো এর ডাটা বের করার জন্য iloc ব্যাবহার করছি।  
df_students.iloc[0:5]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dan</td>
      <td>10.00</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.50</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro</td>
      <td>9.00</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosie</td>
      <td>16.00</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ethan</td>
      <td>9.25</td>
      <td>49.0</td>
    </tr>
  </tbody>
</table>
</div>



 `iloc[0:5]` আর `loc[0:5]` দুই রকম রেজাল্ট দিচ্ছে। 

loc[0:5] *0*, *1*, *2*, *3*, *4*, and *5* (টোটাল ৬ রো) এর ডাটা শো করছে। অন্য দিকে iloc[0:5] পজিশন বেসিসে প্রথম ৫ রো এর ডাটা শো করছে। এখানে আপার বাউন্ডারি ইনক্লুডেড হয় না।  

***iloc*** এর মাধ্যমে নির্দিষ্ট রো এর নির্দিষ্ট কয়েকটা কলামের ডাটা বের করা যায়। যেমন প্রথম রো এর প্রথম দুই কলামের ডাটা বের করছি নিচের কোড ব্যবহার করে। 



```python
df_students.iloc[0,[1,2]]
```




    StudyHours    10.0
    Grade         50.0
    Name: 0, dtype: object



 **loc** মেথডে কিভাবে কলামের সাথে কাজ করা যায়? 
 
**loc** ব্যাবহার করা হয়ে ইনডেক্স ভ্যালু ব্যাবহার করে কাজ করার জন্য। রো এর পজিশনের সাথে এর কোনো সম্পর্ক নেই। রো গুলো ইন্টিজার ভ্যালু দিয়ে আইডেন্টিফাই করা হয় আর কলাম গুলো আইডেন্টিফাই করা হয় কলামের নাম দিয়ে। 



```python
df_students.loc[0,'Grade']
```




    50.0




```python
df_students.loc[0,'StudyHours']
```




    10.0



 **loc** মেথড ব্যাবহার করে ফিল্টা করা যায়। যেমন কোনো কলামের আন্ডারে থাকা কোনো একটা ভ্যালু দিয়ে পুরো রো এর ডাটা বের করা যায়। 



```python

```


```python
df_students.loc[df_students['Name']=='Aisha']
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.0</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>



 **loc** ছাড়ার ডাটাফ্রেমের সিম্পল ফিল্টারিং করেও এই কাজ করা যায়। 


```python
df_students[df_students['Name']=='Aisha']
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.0</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>



এছাড়া **query** মেথড ব্যবহার করেই একই কাজটা করা যায়। 


```python
df_students.query('Name=="Aisha"')
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.0</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>





পান্ডায় একই রেজাল্ট বিভিন্ন মেথড ব্যাবহার করে পাওয়া যায়। কলাম কে ডাটা ফ্রেমের এক্টা প্রোপার্টি হিসেবে ব্যাবহার করেও সেম কাজ টা করা যায়। এখানে .name কলামের নাম। 


```python
df_students[df_students.Name == 'Aisha']
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.0</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>



### কোনো একটা ফাইল থেকে ডাটাফ্রেম লোড করে নেয়ার পদ্ধতি। 



```python
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv
df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')
df_students.head()
```

    --2023-11-24 13:21:13--  https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv
    Loaded CA certificate '/etc/ssl/certs/ca-certificates.crt'
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 2606:50c0:8001::154, 2606:50c0:8003::154, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 322 [text/plain]
    Saving to: ‘grades.csv’
    
    grades.csv          100%[===================>]     322  --.-KB/s    in 0s      
    
    2023-11-24 13:21:19 (18.4 MB/s) - ‘grades.csv’ saved [322/322]
    





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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dan</td>
      <td>10.00</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.50</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro</td>
      <td>9.00</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosie</td>
      <td>16.00</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ethan</td>
      <td>9.25</td>
      <td>49.0</td>
    </tr>
  </tbody>
</table>
</div>



Tডাটাফ্রেমের **read_csv**  মেথড ব্যাবহার করে সিএসভি ফর্মেটের ডাটা লোড করে যায়। 


### মিসিং ভ্যালু হ্যান্ডেল করা

ডাটা সাইন্টিস্ট দের একটা কমন সমস্যা হলো ডাটাবেইজে মিসিং ভ্যালু। 


```python
#ডাটা ফ্রেমে কোনো ডাটা মিসিং আছে কিনা দেখে নিই
df_students.isnull()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



বড় ডাটা সেটে প্রত্যেক রো আর কলামে ডাটা মিসিং আছে কিনা দেখা সময় সাপেক্ষ হবে বলে আমরা প্রত্যেক কলামে কত গুলো ডাটা মিসিং আছে তা দেখার জন্য নিচের কোড ব্যাবহার করছি



```python
df_students.isnull().sum()
```




    Name          0
    StudyHours    1
    Grade         2
    dtype: int64



এখানে দেখতে পাচ্ছি স্টাডিআওয়ারে একটা ভ্যালু আর গ্রেড কলামে দুইটা ভ্যালু মিসিং আছে। 
ঠিক কোন রো তে ডটা মিসিং আছে সেটা দেখার জন্য আমরা ডাটা ফিল্টার করতে পারি। 



```python
df_students[df_students.isnull().any(axis=1)]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>Bill</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Ted</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**NaN** (*not a number*) দেখে বোঝা যায় ঐ স্থানে ডাটা মিসিং আছে। 

মিসিং ভ্যালু তো পেলাম, এখন কি করব? 


একটা কমন পদ্ধতি হলো যেখানে ডাটা মিসিং আছে সেই কলামের একটা এভারেজ ভ্যালু ইনপুট করা, এজন্য **fillna** মেথড ব্যাবহার করতে পারি আমরা।


```python
df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dan</td>
      <td>10.000000</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.500000</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro</td>
      <td>9.000000</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosie</td>
      <td>16.000000</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ethan</td>
      <td>9.250000</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vicky</td>
      <td>1.000000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frederic</td>
      <td>11.500000</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jimmie</td>
      <td>9.000000</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rhonda</td>
      <td>8.500000</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Giovanni</td>
      <td>14.500000</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Francesca</td>
      <td>15.500000</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rajab</td>
      <td>13.750000</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Naiyana</td>
      <td>9.000000</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kian</td>
      <td>8.000000</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Jenny</td>
      <td>15.500000</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Jakeem</td>
      <td>8.000000</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Helena</td>
      <td>9.000000</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ismat</td>
      <td>6.000000</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Anila</td>
      <td>10.000000</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Skye</td>
      <td>12.000000</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Daniel</td>
      <td>12.500000</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.000000</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Bill</td>
      <td>8.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Ted</td>
      <td>10.413043</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



যদি ডাটা বেইসে সঠিক ডাটা ইনপুট না করলে কাজ হবে না এমন হয় তাহলে পুরো রো টা ছাটাই করে ফেলা লাগবে, এজন্য **dropna**  মেথড ব্যবহার করব আমরা । 


```python
df_students = df_students.dropna(axis=0, how='any')
df_students
```

### ডাটাফ্রেমের ডাটা এক্সপ্লোর করছি

ডাটা ক্লিনিং শেষ হয়েছে। এখন আমরা ডাটা এক্সপ্লোর করে দেখেতে পারি।



```python
# কলাম নেইম কে ইনডেক্স হিসেবে ব্যবহার করে কলামের এভারেজ বের করছি। 
mean_study = df_students['StudyHours'].mean()

# কলাম কে ডাটা ফ্রেমের প্রোপার্টি হিসেবে ব্যাবহার করে এর এভারেজ বের করছি। 
mean_grade = df_students.Grade.mean()

#  এভারেজ ভ্যালু গুলোকে প্রিন্ট করে নিচ্ছি 
print('Average weekly study hours: {:.2f}\nAverage grade: {:.2f}'.format(mean_study, mean_grade))
```

    Average weekly study hours: 10.41
    Average grade: 49.18


ডাটাফ্রেমের কোন স্টুডেন্ট গুলো এভারেজ ভ্যালুর চেয়ে বেশি ক্ষন পড়েছে সেটা বের করার জন্য


```python
# যারা এভারেজ আওয়ারের চেয়ে বেশি ক্ষন পড়েছে
df_students[df_students.StudyHours > mean_study]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.50</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosie</td>
      <td>16.00</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frederic</td>
      <td>11.50</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Giovanni</td>
      <td>14.50</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Francesca</td>
      <td>15.50</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rajab</td>
      <td>13.75</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Jenny</td>
      <td>15.50</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Skye</td>
      <td>12.00</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Daniel</td>
      <td>12.50</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.00</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>



এই রেজাল্টটাই নিজে একটা ডাটা ফ্রেম। একে অন্যান্য ডাটাফ্রেমের মতই ব্যবহার করা যায়। 

যারা এভারেজ আওয়ারের বেশি পড়েছে তাদের এভারেজ স্কোর বের করছি নতুন পাওয়া ডাটাবেইজ থেকে। 


```python
# ওদের গ্রেড কত ?

df_students[df_students.StudyHours > mean_study].Grade.mean()
```




    66.7



ধরি পাস করার মার্ক হলো ৬০। স্টুডেন্টরা পাস করেছে কিনা সেটা আরেক্টা কলামে প্রিন্ট করে সেটাকে ডাটাফ্রেমে সেট করতে চাচ্ছি। 

আমরা একটা পান্ডাস **Series** তৈরি করব যেটায় পাস ফেইল ইন্ডিকেটর থাকবে। এবং নতুন কলামকে আগের ডাটাফ্রেমে এড করে নিব। 


```python
passes  = pd.Series(df_students['Grade'] >= 60)
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)

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
      <td>10.000000</td>
      <td>50.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.500000</td>
      <td>50.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro</td>
      <td>9.000000</td>
      <td>47.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rosie</td>
      <td>16.000000</td>
      <td>97.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ethan</td>
      <td>9.250000</td>
      <td>49.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vicky</td>
      <td>1.000000</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frederic</td>
      <td>11.500000</td>
      <td>53.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jimmie</td>
      <td>9.000000</td>
      <td>42.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rhonda</td>
      <td>8.500000</td>
      <td>26.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Giovanni</td>
      <td>14.500000</td>
      <td>74.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Francesca</td>
      <td>15.500000</td>
      <td>82.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rajab</td>
      <td>13.750000</td>
      <td>62.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Naiyana</td>
      <td>9.000000</td>
      <td>37.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kian</td>
      <td>8.000000</td>
      <td>15.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Jenny</td>
      <td>15.500000</td>
      <td>70.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Jakeem</td>
      <td>8.000000</td>
      <td>27.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Helena</td>
      <td>9.000000</td>
      <td>36.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ismat</td>
      <td>6.000000</td>
      <td>35.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Anila</td>
      <td>10.000000</td>
      <td>48.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Skye</td>
      <td>12.000000</td>
      <td>52.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Daniel</td>
      <td>12.500000</td>
      <td>63.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.000000</td>
      <td>64.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Bill</td>
      <td>8.000000</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Ted</td>
      <td>10.413043</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



ডাটাফ্রেম গুলো টেবুলার ডাটার জন্য তৈরি করা। যেকোনো রিলেশনাল ডাটাবেইজের মত এতে ডাটা এনালাইসিস করা যাবে। যেমন গ্রুপ করা যাবে, এগ্রিগেট করা যাবে। 

যেমন  **groupby** মেথড ব্যাবহার করে যারা পাস করেছে তাদের আলাদা ডাটা ফ্রেম তৈরি করা যাবে।  


```python
print(df_students.groupby(df_students.Pass).Name.count())
```

    Pass
    False    17
    True      7
    Name: Name, dtype: int64


ডাটাবেইজে পাস ফেল এর গ্রুপ করে এদের স্টাডি আওয়ার আর গ্রেডের এভারেজ বের করা যাবে। 


```python
print(df_students.groupby(df_students.Pass)[['StudyHours', 'Grade']].mean())
```

           StudyHours      Grade
    Pass                        
    False     8.83312  38.000000
    True     14.25000  73.142857




ডাটাফ্রেম বিভিন্ন ভাবে ম্যানিপুলেট করা যায়। আমরা যদি চাই ম্যানুপুলেট করার পর আগের ডাটাফ্রেমে ভ্যালুগুলো রিপ্লেস হবে তাহলে ওপারেশনের সময় ভ্যারিয়েবলের জায়গায় আগের ডাটা ফ্রেমের নাম দিতে হবে। নাহলে নতুন একটা ডাটাফ্রেম তৈরি হবে।


```python
# গ্রেড গুলোকে সর্ট করে নতুন ডাটাফ্রেম তৈরি করা হচ্ছে
df_students = df_students.sort_values('Grade', ascending=False)

# ডাটাফ্রেম দেখে নিই
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
      <th>3</th>
      <td>Rosie</td>
      <td>16.000000</td>
      <td>97.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Francesca</td>
      <td>15.500000</td>
      <td>82.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Giovanni</td>
      <td>14.500000</td>
      <td>74.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Jenny</td>
      <td>15.500000</td>
      <td>70.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Aisha</td>
      <td>12.000000</td>
      <td>64.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Daniel</td>
      <td>12.500000</td>
      <td>63.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Rajab</td>
      <td>13.750000</td>
      <td>62.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Frederic</td>
      <td>11.500000</td>
      <td>53.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Skye</td>
      <td>12.000000</td>
      <td>52.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joann</td>
      <td>11.500000</td>
      <td>50.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Dan</td>
      <td>10.000000</td>
      <td>50.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ethan</td>
      <td>9.250000</td>
      <td>49.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Anila</td>
      <td>10.000000</td>
      <td>48.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pedro</td>
      <td>9.000000</td>
      <td>47.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Jimmie</td>
      <td>9.000000</td>
      <td>42.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Naiyana</td>
      <td>9.000000</td>
      <td>37.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Helena</td>
      <td>9.000000</td>
      <td>36.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ismat</td>
      <td>6.000000</td>
      <td>35.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Jakeem</td>
      <td>8.000000</td>
      <td>27.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rhonda</td>
      <td>8.500000</td>
      <td>26.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kian</td>
      <td>8.000000</td>
      <td>15.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Vicky</td>
      <td>1.000000</td>
      <td>3.0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Bill</td>
      <td>8.000000</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Ted</td>
      <td>10.413043</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### সামারি

পান্ডাস আর নামপাই দিয়ে বিভিন্ন ভাবে ডাটাবেইজ লোড করে ডাটা ম্যানিপুলেশন করা যায়, আমরা তার বেশ কিছু বিষয়ের ধারনা নিলাম এই নোটবুকে। 

# ধন্যবাদ


```python

```
