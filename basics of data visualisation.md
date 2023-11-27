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




![png](output_1_2.png)




এই ডাটা গুলোর মিন মিডিয়ান মোড সেন্টারে ছিলো, আর ডাটা সেখান থেকে সিমেট্রিক্যালি স্প্রেড করেছে

এখন পড়ার সময়ের ডাটার ডিস্ট্রিবিউশন কেমন দেখে নিচ্ছি


```python
# ভ্যারিয়েবল সিলেক্ট করি
