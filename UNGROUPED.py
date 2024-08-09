import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, norm, iqr
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
st.set_page_config(layout="wide")

def load_data():
    return pd.read_csv('dataset.csv')

st.sidebar.image("logo2.png",caption="EmployeeEcho Insights")
theme_plotly = None 

def load_data():
    return pd.read_csv('dataset.csv')

def calculate_ungrouped_statistics(data):
    # Calculate mean
    mean = data.mean()
    
    # Calculate median
    median = data.median()
    
    # Calculate mode
    mode = data.mode().values[0]
    
    # Calculate standard deviation
    std_dev = data.std()
    
    # Calculate variance
    variance = data.var()
    
    # Calculate skewness
    skewness = skew(data)
    
    # Calculate kurtosis
    kurtosis_value = kurtosis(data)
    
    # Calculate standard error
    std_error = std_dev / np.sqrt(len(data))
    
    # Calculate IQR
    IQR_value = iqr(data)
    
    return mean, mode, median, std_dev, variance, skewness, kurtosis_value, std_error, IQR_value

def main():
    st.title("Ungrouped Data Statistics")

    # Load dataset from CSV
    dataset = load_data()

    # Extract age field
    age_data = dataset['age']

    # Calculate statistics
    mean, mode, median, std_dev, variance, skewness, kurtosis_value, std_error, IQR_value = calculate_ungrouped_statistics(age_data)

    st.subheader("Age-Based Health Analysis: Gender, Weight, Height, and Diabetes Distribution")
    # Print results
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
    style_metric_cards(border_left_color="#e1ff8b",background_color="#222222")

    # Skewness visualization
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
    skew_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray')  # Add gridlines on x-axis
    skew_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray')  # Add gridlines on y-axis

    st.plotly_chart(skew_fig,use_container_width=True)

if __name__ == "__main__":
    main()
