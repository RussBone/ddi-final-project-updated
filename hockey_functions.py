import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
import hockey_rink as hr
import streamlit as st

def plot_hexbin_team(data, team_name):
    rink = hr.NHLRink()
    team_data = data[data["teamCode"] == team_name]

    event_types = ["SHOT", "GOAL", "MISS"]
    titles = ["Shot Hexbin", "Goal Hexbin", "Miss Hexbin"]
    cmaps = ["Blues", "Greens", "Reds"]
    fig, axs = plt.subplots(1,3,figsize=(20, 16))

    for event, title, cmap, ax in zip(event_types, titles, cmaps, axs):
        event_data = team_data[team_data["event"] == event]

        rink.draw(ax=ax, display_range="ozone")
        ax.hexbin(event_data["xCordAdjusted"], event_data["yCordAdjusted"],
                  gridsize=(24, 12), cmap=cmap)
        ax.set_title(f"{team_name} {title}")
    plt.tight_layout()
    st.pyplot(fig)

def plot_hexbin_team_invert_color(data, team_name):
    rink = hr.NHLRink()
    team_data = data[data["teamCode"] == team_name]

    event_types = ["SHOT", "GOAL", "MISS"]
    titles = ["Shot Hexbin", "Goal Hexbin", "Miss Hexbin"]
    cmaps = ["viridis", "viridis", "viridis"]
    fig, axs = plt.subplots(1,3,figsize=(20, 16))

    for event, title, cmap, ax in zip(event_types, titles, cmaps, axs):
        event_data = team_data[team_data["event"] == event]

        rink.draw(ax=ax, display_range="ozone")
        ax.hexbin(event_data["xCordAdjusted"], event_data["yCordAdjusted"],
                  gridsize=(24, 12), cmap=cmap)
        ax.set_title(f"{team_name} {title}")
    plt.tight_layout()
    st.pyplot(fig)

def plot_hexbin_player(data, player_name):
    rink = hr.NHLRink()
    player_data = data[data["shooterName"] == player_name]

    event_types = ["SHOT", "GOAL", "MISS"]
    titles = ["Shot Hexbin", "Goal Hexbin", "Miss Hexbin"]
    cmaps = ["Blues", "Greens", "Reds"]
    fig, axs = plt.subplots(1,3,figsize=(20, 16))

    for event, title, cmap, ax in zip(event_types, titles, cmaps, axs):
        event_data = player_data[player_data["event"] == event]

        rink.draw(ax=ax, display_range="ozone")
        ax.hexbin(event_data["xCordAdjusted"], event_data["yCordAdjusted"],
                  gridsize=(24, 12), cmap=cmap)
        ax.set_title(f"{player_name} {title}")
    plt.tight_layout()
    st.pyplot(fig)

def plot_percent_bar(data, target, title, x_subtitle=None):
    cross = pd.crosstab(data[target], data['event'])
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    ax = cross_pct.plot(kind='bar', stacked=True, figsize=(8,5))
    plt.title(f'Result by {title}')
    plt.xlabel(f"{title}" if not x_subtitle else f"{title} \n {x_subtitle}")
    plt.ylabel('Percentage')
    plt.legend(title='Result')
    for i, (index, row) in enumerate(cross_pct.iterrows()):
        cumulative = 0
        for result in cross_pct.columns:
            value = row[result]
            if value > 0:
                ax.text(
                    i, 
                    cumulative + value / 2, 
                    f'{value:.1f}%', 
                    ha='center', 
                    va='center', 
                    fontsize=8, 
                    color='white' if value > 15 else 'black'  # adjust for readability
                )
            cumulative += value
    plt.tight_layout()
    st.pyplot(plt, use_container_width=False)

def pie_chart(data, target):
    event_count = data[target].value_counts()
    fig, ax = plt.subplots(figsize=(6,4))
    event_count.plot(ax=ax, kind='pie',autopct='%1.1f%%', startangle=90, colors=['tab:green','tab:orange','tab:blue'])
    plt.title('Result per Shot Attempt')
    plt.ylabel(None)
    st.pyplot(fig, use_container_width=False)

def random_row_except(df, exclude_cols):
    if not isinstance(exclude_cols, list):
        exclude_cols = [exclude_cols]

    # Prepare dict for new row
    new_row = {}

    for col in df.columns:
        if col in exclude_cols:
            new_row[col] = np.nan  # or use a placeholder if needed
        else:
            new_row[col] = df[col].sample(1).values[0]

    # Return as a single-row DataFrame
    return pd.DataFrame([new_row])

if __name__ == "__main__":
    data = pd.read_csv("C:\Code\ddi_course\projects\potential_data\data\shots_2024.csv")
    plot_hexbin_player(data, "Alex Ovechkin")
    plt.show()