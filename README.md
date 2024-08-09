# **Ungrouped Data Statistics Program Documentation**

# **PAGE NAME: UNGROUPED.PY**
---
> This program is a Streamlit application designed to analyze ungrouped data, particularly focusing on age data from a dataset. The application calculates various statistical measures and provides visualizations to help understand the distribution and characteristics of the data. Below is an overview of what the program does and the key functionalities it provides.

## **Program Overview**

### **1. Loading the Dataset**

> The program begins by loading the dataset, which is stored in a CSV file named `dataset.csv`. The `load_data()` function is responsible for this task.

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





