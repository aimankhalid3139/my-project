import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Project Dashboard", layout="wide")
st.title("📊 System Efficiency & Predictive Analysis")
st.write("Developed for Academic Project Submission.")

# Data Generation
np.random.seed(42)
hours = np.random.uniform(10, 500, 100)
efficiency = 100 - (hours * 0.05) + np.random.normal(0, 2, 100)
df = pd.DataFrame({"Operating_Hours": hours, "Efficiency_Percent": efficiency})

# Sidebar
hours_input = st.sidebar.slider("Predict Efficiency for Hours:", 10, 600, 250)

# Model
X = df[["Operating_Hours"]]
y = df["Efficiency_Percent"]
model = LinearRegression().fit(X, y)
predicted_eff = model.predict([[hours_input]])[0]

# Display
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Data Log")
    st.dataframe(df.head(10))
    st.metric(label=f"Predicted Efficiency at {hours_input} Hours", value=f"{predicted_eff:.2f}%")
with col2:
    st.subheader("📉 Efficiency Graph")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Operating_Hours", y="Efficiency_Percent", ax=ax, color="blue")
    X_plot = np.linspace(10, 600, 100).reshape(-1, 1)
    ax.plot(X_plot, model.predict(X_plot), color="red", linestyle="--")
    st.pyplot(fig)
    fig.savefig("efficiency_graph.png", dpi=300)