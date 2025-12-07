import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_class_distribution(data, target_column):
    counts = data[target_column].astype(str).value_counts()
    fig = go.Figure(data=[go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color = '#92c5de'
    )])
    fig.update_layout(
        title=dict(
            text="Class Distribution plot",
            font=dict(
                family="Inter, sans-serif",  
                size=14,
                color="white"
            ),
            x=0.1),
        plot_bgcolor='rgba(0,0,0,0)',   
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title="Count",
            tickfont=dict(color='white'),          
            titlefont=dict(color='white'),        
            gridcolor='lightgray',
        ),
        xaxis=dict(
            title="Class",
            tickfont=dict(color='white'),          
            titlefont=dict(color='white'),        
            gridcolor='lightgray',
            type='category'             
        ),
        margin=dict(l=60, r=10, t=80, b=60)
    )
    return fig 

# #for numeric features
def plot_correlation_matrix(data):
    numeric_data = data.select_dtypes(include='number')
    corr = numeric_data.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title='corr',
            thickness=15,   
            tickfont=dict(color='white'),
            titlefont=dict(color='white')
        )
    ))
    fig.update_layout(
        title=dict(
            text="Pearsons correlation Matrix",
            font=dict(
                family="Inter, sans-serif",
                size=14,
                color="white"
            ),
            x=0.1
        ),
        xaxis=dict(
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
            gridcolor='lightgray',
            side='bottom'
        ),
        yaxis=dict(
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
            gridcolor='lightgray'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=0, t=80, b=60)
    )
    return fig

def return_missing(dataset):
    total_cells = dataset.size
    nan_cells = dataset.isna().sum().sum()
    return (nan_cells / total_cells) * 100






# #analysis if data might be skewed
# def plot_features_distribution(data, target):
#     features = data.drop(columns=[target])
#     features.hist(bins=20, figsize=(15, 15), color = 'skyblue')
#     plt.suptitle("Feature Distributions")
#     plt.show()

# #plot categorical features distribution with respect to target
# def plot_categorical_features_distribution(data, target):
#     categorical_data = data.select_dtypes(include='object')
#     n_cols = 3
#     n_rows = (len(categorical_data.columns) + n_cols - 1) // n_cols

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
#     axes = axes.flatten()

#     for i, column in enumerate(categorical_data.columns):
#         sns.countplot(x=column, data=data, hue=target, ax=axes[i])
#         axes[i].set_title(f"Distribution of {column} by {target}")
#         axes[i].set_xlabel(column)
#         axes[i].set_ylabel("Count")

#     for j in range(i+1, len(axes)):
#         axes[j].set_visible(False)

#     plt.tight_layout()
#     plt.show()

