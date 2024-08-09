# **Ungrouped Data Statistics Program Documentation**
![6](https://github.com/user-attachments/assets/be233a50-02bd-4bc4-a44e-e61337141cfa)
![1](https://github.com/user-attachments/assets/b9c133ea-1157-4323-9302-6ff9cba088a4)
![4](https://github.com/user-attachments/assets/6affa8d9-7ee1-446e-a356-a275872d9dfc)
![3](https://github.com/user-attachments/assets/609f490e-1dcd-4ec2-830c-261eea39bf17)

# **PAGE NAME: UNGROUPED.PY**
---
## **Program Overview**
> This program is a Streamlit application designed to analyze ungrouped data, particularly focusing on age data from a dataset. The application calculates various statistical measures and provides visualizations to help understand the distribution and characteristics of the data. Below is an overview of what the program does and the key functionalities it provides.

> The program begins by importing necessary libraries, including:

- **Streamlit**: For creating the web application interface.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **SciPy.stats**: For statistical calculations like skewness and kurtosis.
- **Plotly**: For creating interactive visualizations.
- **Streamlit Extras**: For additional UI enhancements.

```python
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
```
### **1. Loading the Dataset**

> The program continues by loading the dataset, which is stored in a CSV file named `dataset.csv`. The `load_data()` function is responsible for this task.

```python
def load_data():
    return pd.read_csv('dataset.csv')
```


### **2 Statistical Calculations**
> The core functionality of the program revolves around calculating key statistical measures for ungrouped data. These include:
- Mean: The average of the data.
- Median: The middle value of the data.
- Mode: The most frequently occurring value in the data.
- Standard Deviation: A measure of the amount of variation in the data.
- Variance: The square of the standard deviation.
- Skewness: A measure of the asymmetry of the distribution.
- Kurtosis: A measure of the "tailedness" of the distribution.
- Standard Error: An estimate of the standard deviation of the sample mean.
- Interquartile Range (IQR): A measure of statistical dispersion.
- These calculations are performed using the calculate_ungrouped_statistics(data) function.

```python


def calculate_ungrouped_statistics(data):
    mean = data.mean()
    median = data.median()
    mode = data.mode().values[0]
    std_dev = data.std()
    variance = data.var()
    skewness = skew(data)
    kurtosis_value = kurtosis(data)
    std_error = std_dev / np.sqrt(len(data))
    IQR_value = iqr(data)
    return mean, mode, median, std_dev, variance, skewness, kurtosis_value, std_error, IQR_value
```

### **3. Displaying Results**
> The program utilizes Streamlit’s metric function to display the calculated statistics in a visually appealing format. The metrics are organized into columns for better readability.

```python
a1,a2,a3=st.columns(3) 
b1,b2,b3=st.columns(3)   
c1,c2,c3=st.columns(3) 

a1.metric("Mean (Ungrouped Data):", f"{mean:.2f}")
a2.metric("Mode (Ungrouped Data):", f"{mode:.2f}")
a3.metric("Median (Ungrouped Data):", f"{median:.2f}")
b1.metric("Standard Deviation (Ungrouped Data):", f"{std_dev:.2f}")
b2.metric("Variance (Ungrouped Data):", f"{variance:.2f}")
b3.metric("Skewness (Ungrouped Data):", f"{skewness:.2f}")
c1.metric("Kurtosis (Ungrouped Data):", f"{kurtosis_value:.2f}")
c2.metric("Standard Error (Ungrouped Data):", f"{std_error:.2f}")
c3.metric("Interquartile Range (IQR) (Ungrouped Data):", f"{IQR_value:.2f}")
```

### **4. Visualizing Skewness**
> To provide a visual representation of skewness, the program plots the normal distribution curve based on the mean and standard deviation of the data. This is done using Plotly, a graphing library, which allows for interactive visualizations.

```python
x = np.linspace(age_data.min(), age_data.max(), 100)
p = norm.pdf(x, mean, std_dev)
skew_fig = px.line(x=x, y=p, labels={'x': 'Age', 'y': 'Probability Density'})
skew_fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Normal Distribution'))
skew_fig.update_layout(title="Skewness Visualization", xaxis_title="Age", yaxis_title="Probability Density",
                    plot_bgcolor='rgba(0,0,0,0)',  # Set background transparency
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))  # Adjust legend position
skew_fig.add_annotation(
    x=x[0],
    y=p[0],
    text=f"Skewness: {skewness:.2f}",
    showarrow=False,
    font=dict(color="red", size=12)
)
```

### **5. Streamlit Setup**
> Finally, the program is wrapped in a main() function, which is the entry point of the Streamlit application. The page layout is set to wide, and an image is added to the sidebar for branding.
```python
st.set_page_config(layout="wide")
st.sidebar.image("logo2.png",caption="EmployeeEcho Insights")

def main():
    st.title("Ungrouped Data Statistics")
    dataset = load_data()
    age_data = dataset['age']
    mean, mode, median, std_dev, variance, skewness, kurtosis_value, std_error, IQR_value = calculate_ungrouped_statistics(age_data)
    # Display metrics and visualizations here...
    st.plotly_chart(skew_fig,use_container_width=True)

if __name__ == "__main__":
    main()
```

# **Grouped Data Statistics Program Documentation**

This Streamlit application analyzes grouped data, specifically focusing on age data from a dataset. It calculates various statistical measures for grouped data and provides visualizations to help users understand the distribution and characteristics of the data.

## **Program Overview**

### **1. Importing Libraries**

The program begins by importing necessary libraries, including:

- **Streamlit**: For creating the web application interface.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **SciPy.stats**: For statistical calculations like skewness and kurtosis.
- **Plotly**: For creating interactive visualizations.
- **Streamlit Extras**: For additional UI enhancements.

```python
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
```
### **2. Configuring the Streamlit Page**
> The application sets the page layout to wide using st.set_page_config() to utilize the full screen width.

```python
st.set_page_config(layout="wide")
```
### **3. Loading the Dataset**
> The load_data() function is responsible for loading the dataset from a CSV file named dataset.csv.

```python
def load_data():
    return pd.read_csv('dataset.csv')
```

### **4. Sidebar Image**
> An image is added to the sidebar for branding purposes, with the caption "EmployeeEcho Insights".

```python
st.sidebar.image("logo2.png", caption="EmployeeEcho Insights")
```
### **5. Loading Custom CSS**
> The application loads a custom CSS file to style the page. This CSS can include various customizations like colors, fonts, and layout adjustments.

```python
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
```
### **6. Calculating Age Intervals**
> The calculate_age_intervals(max_age) function generates age intervals and corresponding labels for binning the age data. This function creates intervals of 10 years.

```python
def calculate_age_intervals(max_age):
    intervals = np.arange(0, max_age + 11, 10)  # Adjusted to include maximum age
    labels = [f'{i}-{i+10}' for i in range(0, max_age + 1, 10)]  # Adjusted to include maximum age
    return intervals, labels
```
### **7. Calculating Grouped Data Statistics**
> The calculate_grouped_statistics(freq_table) function calculates key statistical measures for the grouped data:

- Mean: The average value.
- Mode: The most frequent interval.
- Median: The middle interval.
- Variance: A measure of dispersion.
- Standard Deviation: The square root of variance.
- Skewness: The asymmetry of the distribution.
- Kurtosis: The "tailedness" of the distribution.
- Interquartile Range (IQR): The range between the first and third quartiles.
- Standard Error: An estimate of the standard deviation of the sample mean.
- The function calculates these statistics by first converting age intervals to numerical midpoints, then applying statistical formulas.

```python
def calculate_grouped_statistics(freq_table):
    # Convert 'Age Interval' to numerical midpoints
    def get_midpoint(interval):
        start, end = map(int, interval.split('-'))
        return (start + end) / 2
    
    # Apply midpoint function and ensure numerical type
    freq_table['Midpoint'] = freq_table['Age Interval'].apply(get_midpoint).astype(float)

    # Calculate cumulative frequency
    freq_table['Cumulative Frequency'] = freq_table['Frequency'].cumsum()

    # Calculate mean
    mean_grouped = (freq_table['Midpoint'] * freq_table['Frequency']).sum() / freq_table['Frequency'].sum()

    # Calculate mode class
    mode_class = freq_table.loc[freq_table['Frequency'].idxmax()]['Age Interval']
    mode_class_start, mode_class_end = map(int, mode_class.split('-'))
    mode_freq = freq_table.loc[freq_table['Age Interval'] == mode_class, 'Frequency'].values[0]
    cumulative_freq_before = freq_table['Frequency'].cumsum().loc[freq_table['Age Interval'] == mode_class].values[0] - mode_freq
    mode = mode_class_start + ((mode_freq - cumulative_freq_before) / (2 * mode_freq)) * (mode_class_end - mode_class_start)

    # Calculate median
    total_freq = freq_table['Frequency'].sum()
    cumulative_freq = freq_table['Cumulative Frequency']
    median_class = freq_table.loc[cumulative_freq >= (total_freq / 2)].iloc[0]['Age Interval']
    median_class_start, median_class_end = map(int, median_class.split('-'))
    median_freq = freq_table.loc[freq_table['Age Interval'] == median_class, 'Frequency'].values[0]
    cumulative_freq_before = cumulative_freq[freq_table['Age Interval'] == median_class].values[0] - median_freq
    median = median_class_start + ((total_freq / 2 - cumulative_freq_before) / median_freq) * (median_class_end - median_class_start)

    # Calculate variance
    variance = ((freq_table['Midpoint'] - mean_grouped)**2 * freq_table['Frequency']).sum() / freq_table['Frequency'].sum()
    std_dev = np.sqrt(variance)

    # Calculate skewness
    skewness_grouped = skew(freq_table['Midpoint'].repeat(freq_table['Frequency']))

    # Calculate kurtosis
    kurtosis_value = kurtosis(freq_table['Midpoint'].repeat(freq_table['Frequency']))

    # Calculate IQR
    Q1 = freq_table.loc[freq_table['Frequency'].cumsum() >= (total_freq * 0.25)].iloc[0]['Midpoint']
    Q3 = freq_table.loc[freq_table['Frequency'].cumsum() >= (total_freq * 0.75)].iloc[0]['Midpoint']
    IQR = Q3 - Q1

    # Calculate standard error
    standard_error = std_dev / np.sqrt(freq_table['Frequency'].sum())

    return mean_grouped, mode, mode_class, median, skewness_grouped, kurtosis_value, IQR, std_dev, standard_error, median_class, variance
```


### **8. Main Application Function**
> The main() function serves as the entry point of the application. It performs the following steps:

- Load the dataset: Reads the data from the CSV file.
- Calculate age intervals: Bins the age data into intervals.
- Create frequency table: Counts the frequency of data points in each interval.
- Filter frequency table: Removes intervals with zero frequency.
- Calculate statistics: Uses the calculate_grouped_statistics() function to compute the statistical measures.
- Display statistics: Metrics like mean, mode, median, and others are displayed using Streamlit's metric function.
- Visualize skewness: Plots the skewness of the data using Plotly.
- Display frequency table: Shows the frequency table with cumulative frequency.

```python
def main():
    st.title("Grouped Data Statistics")

    # Load dataset from CSV
    dataset = load_data()

    # Calculate age intervals and frequencies
    max_age = dataset['age'].max()
    intervals, labels = calculate_age_intervals(max_age)
    dataset['age_intervals'] = pd.cut(dataset['age'], bins=intervals, labels=labels, right=False)
    freq_table = dataset['age_intervals'].value_counts().reset_index()
    freq_table.columns = ['Age Interval', 'Frequency']
    freq_table['Age Interval'] = pd.Categorical(freq_table['Age Interval'], categories=labels, ordered=True)
    freq_table = freq_table.sort_values('Age Interval')

    # Filter out age intervals with zero frequency
    freq_table = freq_table[freq_table['Frequency'] > 0]

    # Calculate statistics
    mean_grouped, mode, mode_class, median_grouped, skewness_grouped, kurtosis_value, IQR, std_dev, standard_error, median_class, variance = calculate_grouped_statistics(freq_table)

    st.subheader("Age-Based Health Analysis: Gender, Weight, Height, and Diabetes Distribution")
    
    # Print results
    a1, a2, a3 = st.columns(3) 
    b1, b2, b3 = st.columns(3)   
    c1, c2, c3 = st.columns(3) 
    d1, d2 = st.columns(2)
    st.subheader("Grouped Data Statistics")
    a1.metric("Mean (Grouped Data):", f"{mean_grouped:.2f}")
    a2.metric("Mode (Grouped Data):", f"{mode:.2f}")
    a3.metric("Mode Class (Grouped Data):", mode_class)
    b1.metric("Median (Grouped Data):", f"{median_grouped:.2f}")
    b2.metric("Median Class (Grouped Data):", median_class)
    b3.metric("Skewness (Grouped Data):", f"{skewness_grouped:.2f}")
    c1.metric("Kurtosis (Grouped Data):", f"{kurtosis_value:.2f}")
    c2.metric("Interquartile Range (IQR) (Grouped Data):", f"{IQR:.2f}")
    c3.metric("Variance (Grouped Data):", f"{variance:.2f}")
    d1.metric("Standard Deviation (Grouped Data):", f"{std_dev:.2f}")
    d2.metric("Standard Error (Grouped Data):", f"{standard_error:.2f}")
    style_metric_cards(border_left_color="#e1ff8b", background_color="#222222")
```
### **9 Skewness visualization**
```python
    x = np.linspace(dataset['age'].min(), dataset['age'].max(), 100)
    p = norm.pdf(x, mean_grouped, std_dev)
    skew_fig = px.line(x=x, y=p, labels={'x': 'Age', 'y': 'Probability Density'})
    skew_fig.add_trace(go.Scatter(x=x, y=p, mode='lines', name='Normal Distribution'))
    skew_fig.update_layout(title="Skewness Visualization", xaxis_title="Age", yaxis_title="Probability Density",
                           plot_bgcolor='rgba(0,0,0,0)',  # Set background transparency
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))  # Adjust legend position
    skew_fig.add_annotation(
        x=x[0],
        y=p[0],
        text=f"Skewness: {skewness_grouped:.2f}",
        showarrow=False,
        font=dict(color="red", size=12)
    )
    skew_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')  # Add gridlines on x-axis
    skew_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')  # Add gridlines on y-axis

    st.plotly_chart(skew_fig, use_container_width=True)

    # Display frequency table with cumulative frequency
    st.dataframe(freq_table)
```

### ***10 run application**
```python
if __name__ == "__main__":
    main()
```

# **PAGE NAME: COMPARISON.PY**


### **1. Importing Libraries**

```python
import streamlit as st  # Imports the Streamlit library, which is used to build interactive web applications.
import pandas as pd  # Imports the Pandas library for data manipulation and analysis.
import plotly.graph_objects as go  # Imports Plotly's graph objects module to create interactive visualizations.

```
```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
st.set_page_config(layout="wide")
```
### **2. Loading Data**


```python
def load_data():
    return pd.read_csv('dataset.csv')
```

### **3. Plotting Age vs Various Health Metrics**

```python
def plot_age_vs_fields(df):
    # Create a figure
    fig = go.Figure()

    # Plot age vs weight
    fig.add_trace(go.Bar(
        x=df['age'],
        y=df['Weight'],
        name='Weight',
        marker_color='blue'
    ))

    # Plot age vs height
    fig.add_trace(go.Bar(
        x=df['age'],
        y=df['Height'],
        name='Height',
        marker_color='green'
    ))

    # Plot age vs diabetes status
    df['Diabetes_Num'] = df['Diabetes'].map({'Yes': 1, 'No': 0})
    fig.add_trace(go.Bar(
        x=df['age'],
        y=df['Diabetes_Num'],
        name='Diabetes',
        marker_color='red'
    ))

    # Plot age vs sugar level
    fig.add_trace(go.Bar(
        x=df['age'],
        y=df['Sugar_Level'],
        name='Sugar Level',
        marker_color='orange'
    ))

    # Plot age vs gender
    df['Gender_Num'] = df['Gender'].map({'Male': 1, 'Female': 0})
    fig.add_trace(go.Bar(
        x=df['age'],
        y=df['Gender_Num'],
        name='Gender',
        marker_color='purple'
    ))

    # Update layout
    fig.update_layout(
        barmode='stack',
        title='age vs Various Health Metrics',
        xaxis_title='age',
        yaxis_title='Value',
        plot_bgcolor='rgba(0,0,0,0)',  # Set background transparency
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),  # Adjust legend position
        autosize=True,
        height=600
    )

    # Show plot
    st.plotly_chart(fig, use_container_width=True)

```

### **4. Main Function**

```python
def main():
    st.title("Health Data Analysis")

    # Load dataset from CSV
    df = load_data()

    # Show age distribution by gender with stacked bar
    st.subheader("age Distribution by Gender")
    age_gender_dist_fig = go.Figure()
    gender_dist = df.groupby(['age', 'Gender']).size().reset_index(name='Count')
    for gender in df['Gender'].unique():
        gender_data = gender_dist[gender_dist['Gender'] == gender]
        age_gender_dist_fig.add_trace(go.Bar(
            x=gender_data['age'],
            y=gender_data['Count'],
            name=gender,
            marker_color='blue' if gender == 'Male' else 'pink'
        ))

    age_gender_dist_fig.update_layout(
        barmode='stack',
        title='age Distribution by Gender',
        xaxis_title='age',
        yaxis_title='Count',
        plot_bgcolor='rgba(0,0,0,0)',  # Set background transparency
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),  # Adjust legend position
        autosize=True,
        height=600
    )
    st.plotly_chart(age_gender_dist_fig, use_container_width=True)

    # Plot age vs various fields
    st.subheader("age vs Various Health Metrics")
    plot_age_vs_fields(df)

if __name__ == "__main__":
    main()
```

**My Contacts**

**WhatsApp**  
+255675839840  
+255656848274

**YouTube**  
[Visit my YouTube Channel](https://www.youtube.com/channel/UCjepDdFYKzVHFiOhsiVVffQ)

**Telegram**  
+255656848274  
+255738144353

**PlayStore**  
[Visit my PlayStore Developer Page](https://play.google.com/store/apps/dev?id=7334720987169992827&hl=en_US&pli=1)

**GitHub**  
[Visit my GitHub](https://github.com/shamiraty/)




