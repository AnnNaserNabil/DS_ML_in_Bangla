# পাইথনের মাধ্যমে ডাটা এক্সপ্লোর করা - রিয়েল ওয়ার্ড ডাটা 

আগের নোটবুকে আমরা স্টুডেন্ট এর গ্রেডের ডাটার কিছু ভিজুয়াল দিক দেখেছি হিস্টোগ্রাম আর বক্স প্লটের মাধ্যমে। এখন তুলনামূলক জটিল কিছু বিষয় দেখব, ডাটাকে পুরোপুরি ভাবে ডেস্ক্রাইব করা থেকে শুরু করে তাদের মধ্যকার কিছু সম্পর্ক দেখব। 


### Real world data distributions

আগে আমরা স্টুডেন্ট এর গ্রেডের স্যাম্পল ডাটা দেখে বোঝার চেষ্টা করছিলাম যে সব স্টুডেন্টের গ্রেডের ডাটা কেমন হবে। এখন আবার শুরু থেকে ডাটাটা দেখি এনালাইসিস করি নতুন ভাবে। 

নিচের কোড রান করে হিস্টোগ্রাম আর বক্স প্লট দেখে নিই স্যাম্পল ডাটা থেকে। 


```python
import pandas as pd
from matplotlib import pyplot as plt

# ডাটা লোড করি

df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')

# কোনো ডাটা মিসিং থাকলে তার রো বাদ দিয়ে দিচ্ছি
df_students = df_students.dropna(axis=0, how='any')

# ৬০ মার্কের বেশি পেয়ে যারা পাস করেছে তাদের নতুন ডাটা ফ্রেমে নিই   
passes  = pd.Series(df_students['Grade'] >= 60)

# ডাটাফ্রেম সেভ করি
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)


# নতুন ডাটাফ্রেম প্রিন্ট নিই
print(df_students)


# রি ইউজেবল ফাংশন নিই
def show_distribution(var_data):
    '''
    This function will make a distribution (graph) and display it
    '''

    # ডাটার স্ট্যাটিস্টিকস 
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

    # দুইটা সাবপ্লটের একটা ফিগার নিই
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # হিস্টোগ্রাম প্লট করি
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # মিন মিডিয়ান মোডের লাইন এড করি
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


show_distribution(df_students['Grade'])
```

             Name  StudyHours  Grade   Pass
    0         Dan       10.00   50.0  False
    1       Joann       11.50   50.0  False
    2       Pedro        9.00   47.0  False
    3       Rosie       16.00   97.0   True
    4       Ethan        9.25   49.0  False
    5       Vicky        1.00    3.0  False
    6    Frederic       11.50   53.0  False
    7      Jimmie        9.00   42.0  False
    8      Rhonda        8.50   26.0  False
    9    Giovanni       14.50   74.0   True
    10  Francesca       15.50   82.0   True
    11      Rajab       13.75   62.0   True
    12    Naiyana        9.00   37.0  False
    13       Kian        8.00   15.0  False
    14      Jenny       15.50   70.0   True
    15     Jakeem        8.00   27.0  False
    16     Helena        9.00   36.0  False
    17      Ismat        6.00   35.0  False
    18      Anila       10.00   48.0  False
    19       Skye       12.00   52.0  False
    20     Daniel       12.50   63.0   True
    21      Aisha       12.00   64.0   True
    Minimum:3.00
    Mean:49.18
    Median:49.50
    Mode:50.00
    Maximum:97.00
    


    /tmp/ipykernel_37346/3847978908.py:63: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_1_2.png)
    



এই ডাটা গুলোর মিন মিডিয়ান মোড সেন্টারে ছিলো, আর ডাটা সেখান থেকে সিমেট্রিক্যালি স্প্রেড করেছে 

এখন পড়ার সময়ের ডাটার ডিস্ট্রিবিউশন কেমন দেখে নিচ্ছি


```python
# ভ্যারিয়েবল সিলেক্ট করি
col = df_students['StudyHours']
# ফাংশন কল করি
show_distribution(col)
```

    Minimum:1.00
    Mean:10.52
    Median:10.00
    Mode:9.00
    Maximum:16.00
    


    /tmp/ipykernel_37346/3847978908.py:63: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_3_2.png)
    


গ্রেডের ডাটার ডিস্ট্রিবিউশন থেকে পড়ার সময়ের ডাটার ডিস্ট্রিবিশন বেশ আলাদা

এখানে দেখা যাচ্ছে হুইস্কার বক্স প্লট শুরু হয়েছে ৬ থেকে, অর্থাত বেশির ভাগ স্টুডেন্টের ফার্স্ট কোয়ার্টার শুরু হয়েছে ৬ এর পর থেকে। মিনিমান সময় মার্ক করা হয়েছে **০** দিয়ে। এটাকে স্ট্যাটিস্টিক্যালি আউটলায়ার বলা যায়।  

আউটলায়ার গুলো টিপিক্যাল ডাটা ডিস্ট্রিবিশনে পরে না, কোনো কারনে ডাটা ইনপুট ভুল হলে ডিস্ট্রিবিউশনের বাইরে গিয়ে পরে যায় ওগুলো , যা ম্যাক্সিমাম ভ্যালুর সাথে সামঞ্জস্যপূর্ণ হয় না। 



```python
# এক্সামিন করার জন্য ভ্যারিয়েবল সিলেক্ট করি
# যারা এক ঘন্টার বেশি পড়েছে তাদের বের করতে চাই
col = df_students[df_students.StudyHours>1]['StudyHours']

# ফাংশন কল করি
show_distribution(col)
```

    Minimum:6.00
    Mean:10.98
    Median:10.00
    Mode:9.00
    Maximum:16.00
    


    /tmp/ipykernel_37346/3847978908.py:63: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_5_2.png)
    


এখানে শেখার জন্য আমরা ১ ঘন্টা পড়ার ভ্যালুটা ট্রু আউটলায়ার হিসেসবে বিবেচনা করে বাদ দিয়েছি, ছোট স্যাম্পল সাইজ হওয়াতে এটার প্রভাব বেশ বড় হতো। ২২ জন ছাত্রের জায়গায় যদি ১০০০ জনের ডাটা আমাদের কাছে থাকতো তাহলে আমরা হয়ত দেখতে পেতাম বেশ কিছু মানুষ থাকতো যারা খুব একটা বেশি সময় পড়ে না। 

আমাদের কাছে যত বেশি ডাটা থাকবে আমাদের স্যাম্পল তত বেশি রিলায়েবল হবে। আউটলায়ার গুলো এবোভ আর বিলো পার্সেন্টাইলের বেশি বা কমে থাকে সব সময়। নিচের কোড পান্ডায় **quantile** ফাংশন ব্যাবহার করে ০.০১% কোয়ান্টাইল ডাটা এক্সক্লুড করে ফেলে।  


```python
# ০.০১তম পার্সেন্টাইল হিসেব করা
q01 = df_students.StudyHours.quantile(0.01)
# ভ্যারিয়েবল সিলেক্ট করি
col = df_students[df_students.StudyHours>q01]['StudyHours']
# ফাংশন কল করি
show_distribution(col)
```

    Minimum:6.00
    Mean:10.98
    Median:10.00
    Mode:9.00
    Maximum:16.00
    


    /tmp/ipykernel_37346/3847978908.py:63: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      fig.show()



    
![png](/assets/SentimentAnalyis/output_7_2.png)
    



> ৯৯% কোয়ান্টাইলের উপর বাউন্ডের আউটলায়ার এলিমিনেট করার জন্য উপার বাউণ্ডে থ্রেশহোল্ড পার্সেন্টাইল ভ্যালু চুজ করতে হবে। 


আউটলায়ার রিমুভ করার পর বক্সপ্লট সব ডাটাকে ৪ কোর্টাইলে শো করে। দেখা যাচ্ছে এখানে ডাটা গ্রেডের ডাটার মত সিমেট্রিক না। কেউ অনেক বেশি সময় পড়েছে আবার কেউ অনেক কম সময় পড়েছে , কেউ এভারেজ সময় নিয়ে পড়েছে। 
ডিস্ট্রিবিউশনের ডেনসিটি টা দেখে নিই। 



```python
def show_density(var_data):
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

# পড়ার সময়ের ডেনসিটি দেখে নিই
show_density(col)
```


    
![png](/assets/SentimentAnalyis/output_9_0.png)
    


এমন ডিস্ট্রিবিউশনকে রাইট স্কিউড ডিস্ট্রিবিউশন বলে। বেশির ভাগ ডাটা ডিস্ট্রিবিউশনের লেফট সাইডে আছে। ডান পাশে একটা বিশাল লেজের মত তৈরি হয়েছে বড় ডাটা গুলো থাকার জন্য আর এটা মিনকে ডান দিকে নিয়ে গেছে। 

#### ভ্যারিয়েন্স মাপা

এখন পর্যন্ত আমরা এটা জানি যে গ্রেড আর পড়ার সময়ের ডাটার মাঝের পোর্শন গুলো কোথায় আছে, এছাড়া আরেকটা বিষয় নিয়ে আমাদের দেখার আছে, ডাটার মধ্যে ভ্যারিয়েবিলিটি কেমন আছে?



আছভ্যারিয়েবিলিটি মাপার জন্য স্ট্যাটিস্টীকস এ ৩ ধরনের মাপ 

- **Range**:  ম্যাক্সিমাম আর মিনিমামের মধ্যে পার্থক্য। রেঞ্জের জন্য কোনো বিল্ট ইন ফাঙ্গশন না থাকলেও  **min** আর **max** ফাঙ্গশন ব্যাবহার করে মাপা যায়। 
- **Variance**:  ডাটা গুলো মিন থেকে কত দূরে আছে তার পরিমাপক ।  **var** ফাংশন ব্যাবহার করে এটা বের করা যায়।
- **Standard Deviation**: ভ্যারিয়েন্সের স্কয়ার রুট। বিল্ট ইন **std** ফাঙ্গশন ব্যাবহার করে বের করা যায়। 


```python
for col_name in ['Grade','StudyHours']:
    col = df_students[col_name]
    rng = col.max() - col.min()
    var = col.var()
    std = col.std()
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(col_name, rng, var, std))
```

    
    Grade:
     - Range: 94.00
     - Variance: 472.54
     - Std.Dev: 21.74
    
    StudyHours:
     - Range: 15.00
     - Variance: 12.16
     - Std.Dev: 3.49



স্ট্যাটিস্টিক্যাল মেজারের মধ্যে স্ট্যান্ডার্ড ডেভিয়েশন বহুল ব্যাবহৃত। এটা ভাটা যে স্কেলে আছে সেই স্কেলেই ডাটার ভ্যারিয়েন্স দেখায়। স্ট্যান্ডার্ড ডেভিয়েশন বেশি মানে হলো ডাটায় ভ্যারিয়েন্স বেশি, ডাটা গুলো মিন থেকে বেশ দূরত্বে রয়েছে। 


নর্মাল ডিস্ট্রিবিউশনে , স্ট্যান্ডার্ড ডেভিয়েশন কিছু নিজস্ব বৈশিষ্ট্য নিয়ে কাজ করে থাকে। নিচের কোড রান করে ডাটার সাথে স্ট্যান্ডার্ড ডেভিয়েশনের রিলেশন দেখার চেষ্টা করি। 


```python
import scipy.stats as stats

# গ্রেডের কলাম নিই
col = df_students['Grade']

# ডেনসিটি বের করি
density = stats.gaussian_kde(col)

# ডেনসিটি প্লট করি
col.plot.density()

# মিন আর স্ট্যান্ডার্ড ডেভিয়েশন মাপি
s = col.std()
m = col.mean()

# একটা স্ট্যান্ডার্ড ডেভিয়েশন এনোটেট করি
x1 = [m-s, m+s]
y1 = density(x1)
plt.plot(x1,y1, color='magenta')
plt.annotate('1 std (68.26%)', (x1[1],y1[1]))

# দুইটা স্ট্যান্ডার্ড ডেভিয়েশন এনোটেট করি
x2 = [m-(s*2), m+(s*2)]
y2 = density(x2)
plt.plot(x2,y2, color='green')
plt.annotate('2 std (95.45%)', (x2[1],y2[1]))

# ৩ স্ট্যান্ডার্ড ডেভিয়েশন এনোটেট করি
x3 = [m-(s*3), m+(s*3)]
y3 = density(x3)
plt.plot(x3,y3, color='orange')
plt.annotate('3 std (99.73%)', (x3[1],y3[1]))

# মিনের লোকেশন দেখি
plt.axvline(col.mean(), color='cyan', linestyle='dashed', linewidth=1)

plt.axis('off')

plt.show()
```


    
![png](/assets/SentimentAnalyis/output_13_0.png)
    


 হরাইজন্টাল লাইন গুলো দিয়ে ডাটার ১,২,৩ স্ট্যান্ডার্ড ডেভিয়েশন মার্ক করা হয়েছে। 


যেকোনো নর্মাল ডিস্ট্রিবিউশনে 
- প্রায় 68.26% ডাটা মিন থেকে এক স্ট্যান্ডার্ড ডেভিয়েশনে থাকে
- প্রায় 95.45% ডাটা মিন থেকে  স্ট্যান্ডার্ড ডেভিয়েশনে থাকে
- প্রায় 99.73% সুইডাটা মিন থেকে তিন স্ট্যান্ডার্ড ডেভিয়েশনে থাকে


যেহেতু আমরা জানি মিন গ্রেড ৪৯.১৮ আর স্ট্যান্ডার্ড ডেভিয়েশন হলো ২১.৭৪, গ্রেডের ডিস্ট্রিবিউশন নর্মাল। আমরা এটা বলতে পারি ৬৮.২৬% ছাত্রের মোটামুটি ২৭.৪৪ থেকে ৭০.৯২ এর মধ্যে গ্রেড পাওয়া উচিত। 

ডাটা ফ্রেমে ডেস্ক্রাইব ফাংশন আছে যেটার মাধ্যমে আমরা এখন পর্যন্ত ব্যবহার করা স্ট্যাটিস্টিক্যাল মেজার গুলো দেখতে পাব। 


```python
df_students.describe()
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
      <th>StudyHours</th>
      <th>Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22.000000</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10.522727</td>
      <td>49.181818</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.487144</td>
      <td>21.737912</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>36.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.000000</td>
      <td>49.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.375000</td>
      <td>62.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16.000000</td>
      <td>97.000000</td>
    </tr>
  </tbody>
</table>
</div>



## ডাটা কম্পেয়ার করা

যেহেতু আমরা ডাটার স্ট্যাটিস্টিক্যাল কিছু মেজার দেখেছি, আমরা এখন ভ্যারিয়েবল গুলোর মধ্যে কোনো সম্পর্ক আছে কিনা তা খুঁজে বের করতে পারব। 


শুরুতেই আউটলায়ার আছে এমন রো বাদ দিয়ে দিই। যাতে ডাটা গুলো টিপিক্যাল ছাত্রের গ্রেড আর পড়ার সময়ের ডাটা গুলো থাকে। 


```python
df_sample = df_students[df_students['StudyHours']>1]
df_sample
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



### নিউমেরিক ও ক্যাটাগরিক্যাল ভ্যালু কম্পেয়ার করি

ডাটায় দুইটা *numeric* ভ্যরিয়েবল (**StudyHours** আর **Grade**) আর দুইটা *categorical* ভ্যারিয়েবল (**Name** আর **Pass**) আছে। শুরুতেই নিউমেরিক **StudyHours**  কলামের সাথে ক্যাটাগরিক্যাল **Pass** কলাম কম্পেয়ার করে দেখি যদি এখানে দৃশ্যমান কোনো রিলেশন আছে কিনা।  

এই কম্পেয়ার করার জন্য আমরা বক্সপ্লট নিই দুইটা কলামের জন্য। 


```python
df_sample.boxplot(column='StudyHours', by='Pass', figsize=(8,5))
```




    <Axes: title={'center': 'StudyHours'}, xlabel='Pass'>




    
![png](/assets/SentimentAnalyis/output_19_1.png)
    


স্টাডি আওয়ারের সাথে পাস কলামের ডাটা কম্পেয়ার করে আমরা দেখতে পারি যে যারা পাস করেছে তার যারা পাস করতে পারি নি তাদের চেয়ে বেশি সময় পড়েছে। 


### নিউমেরিক ভ্যারিয়েবল গুলোর মধ্যে কম্পেয়ার 

এখন দুইটা নিউমেরিক ভ্যালু কম্পেয়ার করি। আমরা বার চার্ট তৈরি করে রিলেশন দেখার চেষ্টা করব। 


```python
# নাম , গ্রেড আর স্টাডি আওয়ারে বার চার্ট তৈরি করি
df_sample.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))
```




    <Axes: xlabel='Name'>




    
![png](/assets/SentimentAnalyis/output_21_1.png)
    


এই চার্টে স্টাডি আওয়ার আর গ্রেডের ডাটা থাকলেও ঠিক অনুমান করা যায় না এদের মধ্যে রিলেশন আছে কিনা, ডাটা গুলো দুইটা আলদা স্কেলে আছে। গ্রেডের স্কেল ৩-৯৭ রেঞ্জে আর স্টাডি আওয়ারের স্কেল ১-১৬। 

দুইটা আলাদা ধরনের ডাটার মধ্যে সম্পর্কে ধারনা পেতে আমরা যে কাজ করতে পারি সেটা হলো ডাটাগুলোকে নর্মালাইজ করে ফেলা। এতে করে ডাটা গুলো এক স্কেলে চলে আসবে। আমরা *MinMax* টেকনিক ব্যাবহার করে এই ডাটা গুলোকে নর্মাল ডিস্ট্রিবিউশনের স্কেলে নিয়ে আসতে পারি।  আপনি নিজে কোড লিখে এই কাজটি করতে পারেন , আবার  **Scikit-Learn** ব্যবহার করেও এই কাজটা করা যায়। 


```python
from sklearn.preprocessing import MinMaxScaler

# স্কেলার অবজেক্ট নিচ্ছি
scaler = MinMaxScaler()

# নাম, গ্রেড আর স্টাডি আওয়ারের ডাটা নিয়ে নতুন ডাটাফ্রেম নিচ্ছি
df_normalized = df_sample[['Name', 'Grade', 'StudyHours']].copy()

# নিউমেরিক কলাম গুলো নর্মালাইজ করি
df_normalized[['Grade','StudyHours']] = scaler.fit_transform(df_normalized[['Grade','StudyHours']])

# নর্মালাইজড ভ্যালু গুলো প্লট করি
df_normalized.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))
```




    <Axes: xlabel='Name'>




    
![png](/assets/SentimentAnalyis/output_23_1.png)
    



ডাটা গুলো যেহেতু নর্মালাইজড করা হয়েছে এখন আমরা সহজেই এক স্কেলে থাকায় তাদের মধ্যকার যেকনো রিলেশন বুঝতে পারব। এই গ্রাফে এক্সাক্টলি ম্যাচ না করলেও আমরা বুঝতে পারছি যে যারা বেশি সময় পড়েছে তারাই বেশি ভাল গ্রেড পেয়েছে।  

তাহলে এটা বলা যায় গ্রেডের সাথে পড়ার সময়ের একটা কো রিলেশন আছে।


```python
df_normalized.Grade.corr(df_normalized.StudyHours)
```




    0.9117666413789677





কো রিলেশনের স্ট্যাটিস্টিক্যাল ভ্যালুর রেঞ্জ -১ থেকে ১ এর মধ্যে থাকে। যা দিয়ে দুই ভ্যারিয়েবলের মধ্যকার সম্পর্কের শক্তি বোঝা যায়। 


স্কেলার প্লট ব্যবহার করেও দুইটা ভ্যারিয়েবলের মাঝে কো রিলেশন আছে কিনা দেখা যায়। 


```python
# Create a scatter plot
df_sample.plot.scatter(title='Study Time vs Grade', x='StudyHours', y='Grade')
```




    <Axes: title={'center': 'Study Time vs Grade'}, xlabel='StudyHours', ylabel='Grade'>




    
![png](/assets/SentimentAnalyis/output_27_1.png)
    


এটা থেকে দেখতে পাচ্ছি যারা বেশি পড়েছে তাদের গ্রেড ভাল এসেছে। 

আমরা এটা আরো ভালভাবে দেখতে পাব যদি একটা রিগ্রেশন লাইন এড করি। 

 **SciPy** প্যাকেজে **stats**  ক্লাস আছে যার **linregress** মেথড ব্যাবহার করে সহজে রিগ্রেশন করা যায়। 


```python
from scipy import stats

#
df_regression = df_sample[['Grade', 'StudyHours']].copy()

# রিগ্রেশনের স্লোপ আর ইন্টারসেপ্ট বের করি
m, b, r, p, se = stats.linregress(df_regression['StudyHours'], df_regression['Grade'])
print('slope: {:.4f}\ny-intercept: {:.4f}'.format(m,b))
print('so...\n f(x) = {:.4f}x + {:.4f}'.format(m,b))

# (mx + b) বের করে  f(x) ক্যালকুলেট করি। 
df_regression['fx'] = (m * df_regression['StudyHours']) + b

# f(x) আর y এর মধ্যে এরর বের করি
df_regression['error'] = df_regression['fx'] - df_regression['Grade']

# স্কেটার প্লট এড করি
df_regression.plot.scatter(x='StudyHours', y='Grade')

# রিগ্রেশন লাইন এড করি
plt.plot(df_regression['StudyHours'],df_regression['fx'], color='cyan')

# প্লট শো করি
plt.show()
```

    slope: 6.3134
    y-intercept: -17.9164
    so...
     f(x) = 6.3134x + -17.9164



    
![png](/assets/SentimentAnalyis/output_29_1.png)
    



```python
# অরিজিনাল x,y, f(x), error ভ্যালু বের করি। 
df_regression[['StudyHours', 'Grade', 'fx', 'error']]
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
      <th>StudyHours</th>
      <th>Grade</th>
      <th>fx</th>
      <th>error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.00</td>
      <td>50.0</td>
      <td>45.217846</td>
      <td>-4.782154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.50</td>
      <td>50.0</td>
      <td>54.687985</td>
      <td>4.687985</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.00</td>
      <td>47.0</td>
      <td>38.904421</td>
      <td>-8.095579</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.00</td>
      <td>97.0</td>
      <td>83.098400</td>
      <td>-13.901600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.25</td>
      <td>49.0</td>
      <td>40.482777</td>
      <td>-8.517223</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11.50</td>
      <td>53.0</td>
      <td>54.687985</td>
      <td>1.687985</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9.00</td>
      <td>42.0</td>
      <td>38.904421</td>
      <td>-3.095579</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.50</td>
      <td>26.0</td>
      <td>35.747708</td>
      <td>9.747708</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14.50</td>
      <td>74.0</td>
      <td>73.628262</td>
      <td>-0.371738</td>
    </tr>
    <tr>
      <th>10</th>
      <td>15.50</td>
      <td>82.0</td>
      <td>79.941687</td>
      <td>-2.058313</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13.75</td>
      <td>62.0</td>
      <td>68.893193</td>
      <td>6.893193</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9.00</td>
      <td>37.0</td>
      <td>38.904421</td>
      <td>1.904421</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8.00</td>
      <td>15.0</td>
      <td>32.590995</td>
      <td>17.590995</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15.50</td>
      <td>70.0</td>
      <td>79.941687</td>
      <td>9.941687</td>
    </tr>
    <tr>
      <th>15</th>
      <td>8.00</td>
      <td>27.0</td>
      <td>32.590995</td>
      <td>5.590995</td>
    </tr>
    <tr>
      <th>16</th>
      <td>9.00</td>
      <td>36.0</td>
      <td>38.904421</td>
      <td>2.904421</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6.00</td>
      <td>35.0</td>
      <td>19.964144</td>
      <td>-15.035856</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10.00</td>
      <td>48.0</td>
      <td>45.217846</td>
      <td>-2.782154</td>
    </tr>
    <tr>
      <th>19</th>
      <td>12.00</td>
      <td>52.0</td>
      <td>57.844698</td>
      <td>5.844698</td>
    </tr>
    <tr>
      <th>20</th>
      <td>12.50</td>
      <td>63.0</td>
      <td>61.001410</td>
      <td>-1.998590</td>
    </tr>
    <tr>
      <th>21</th>
      <td>12.00</td>
      <td>64.0</td>
      <td>57.844698</td>
      <td>-6.155302</td>
    </tr>
  </tbody>
</table>
</div>



### রিগ্রেশন কোএফিসিয়েন্ট ব্যাবহার করে প্রেডিক্ট করি

আমাদের কাছে রিগ্রেশন কোএফিসিয়েন্ট থাকায় এটা ব্যাবহার করে যেকোনো স্টাডি আওয়ারের জন্য গ্রেড প্রেডিক্ট করতে পারি


```python
# ফাংশন ডিফাইন করি
def f(x):
    m = 6.3134
    b = -17.9164
    return m*x + b

study_time = 14

# স্টাডি টাইম প্রেডিকশন ফাইলে নিই  
prediction = f(study_time)

# গ্রেড প্রেডিক্ট 
expected_grade = max(0,min(100,prediction))

# এস্টিমেটেড গ্রেড শো করি
print ('Studying for {} hours per week may result in a grade of {:.0f}'.format(study_time, expected_grade))
```

    Studying for 14 hours per week may result in a grade of 70


## সারসংক্ষেপ

এখন পর্যন্ত আমরা যা যা 

1. আউটলায়ার কি? কিভাবে আউট লায়ার গুলো বাদ দেয়া যায়?
2. ডাটার স্কিউনেস কি?
3. ডাটার স্প্রেডনেস দেখেছি
4.  ডাটার মধ্যে সম্পর্ক দেখেছি। 

## Further Reading

আরো জানতে,

- [NumPy](https://numpy.org/doc/stable/)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib](https://matplotlib.org/contents.html)



```python

```
