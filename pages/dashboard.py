import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import plotly.graph_objs as go
st.set_option('deprecation.showPyplotGlobalUse', False)
from plotly.subplots import make_subplots

st.title("Laptop Prices Analysis")

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest


data = pd.read_csv('featured_data.csv')

columns = ["MRP", "RAM In GB", 'RAM TYPE', "Storage", 'Operating System' , 'Display']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

for col, ax in zip(columns, axes.flatten()):
    sns.violinplot(data=data, y=data['MRP'], ax=ax, inner="box", palette="Set3", cut=2, linewidth=3)
    ax.set_title(col, fontsize=18, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Count', fontsize=14)

fig.suptitle("Univariate Plots", fontsize=24, fontweight='bold', y=1.03)
plt.tight_layout()

# Display the figure in Streamlit
st.pyplot(fig)


def multivariate_plot():
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Ram Size vs Price", "StorageType vs Price" ,"OS vs Price"))

    # Create box plots for Ram Size, Storage, StorageType, and OS
    fig.add_trace(px.box(data, x="RAM In GB", y='MRP').data[0], row=1, col=1)
    fig.add_trace(px.box(data, x='Storage', y='MRP').data[0], row=1, col=2)
    fig.add_trace(px.box(data, x='Operating System', y='MRP').data[0], row=1, col=3)

    # Update the layout
    fig.update_layout(height=600, width=800, showlegend=False, font_color="blue", title="Multivariate plots", title_font_size=30, title_font_color="red", font=dict(size=18, family='Arial'))

    
    # Display the figure in Streamlit
    st.plotly_chart(fig)
   

def app():

    # Add markdown text for univariate plots
    st.markdown("## Univariate Insights")
    st.markdown("""The Above plots display univariate distributions of the dataset\n
* The price, ram_size and storage figures also shows a boxplot represntation and the median of each features.
* The first plot illustrates the distribution of the target variable (price) and highlights the unequal distribution of price,
 indicating the need for data transformation.\n
* The other plots show the count of various columns in the dataset, including Ram Size, Storage, and Operating System; which shows unequal distribution""")


    # Add multivariate plot
    multivariate_plot()
    
    # Add markdown text for multivariate plot
    st.markdown("## Multivariate Insights")
    st.markdown("""
The above box plots depict the relationships between the target variable, price, and various features in the dataset. 
Specifically:\n
* The First plot show the relationship between the Ram Size and price
and shows the price seems to increase with increase in ram size
* The Second Plot shows the relationship between Storage and price
showing a bit of linear relationship. The higher the storage, the higher the price.
* Also the other plots shows relationship between StorageType, OS and price and it shows a high number of outliers.
""")


if __name__ == "__main__":
    app()