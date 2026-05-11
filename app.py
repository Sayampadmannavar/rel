import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random

st.title(
    "Smart Traffic Management Dashboard"
)

traffic_data = {

    "North": random.randint(0, 50),
    "South": random.randint(0, 50),
    "East": random.randint(0, 50),
    "West": random.randint(0, 50)
}

df = pd.DataFrame(
    traffic_data.items(),
    columns=["Road", "Vehicles"]
)

st.write(df)

fig, ax = plt.subplots()

ax.bar(
    df["Road"],
    df["Vehicles"]
)

st.pyplot(fig)

st.success(
    "RL Traffic System Running Successfully"
)