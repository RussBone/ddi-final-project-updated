import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, r2_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from hockey_functions import *
from hockey_rink import *
import joblib
import requests

st.set_page_config(layout="wide")
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)  # 👈 Download the data
    return df
# response = requests.get("http://127.0.0.1:8000/data")
# data = pd.DataFrame(response.json())
df = load_data("data/shots_sample.csv")
data = df.drop(['shotID','homeTeamCode','awayTeamCode','game_id','homeTeamWon','isHomeTeam','location',
                    'playerNumThatDidEvent','playerNumThatDidLastEvent','arenaAdjustedShotDistance','arenaAdjustedXCord',
                    'arenaAdjustedYCord','arenaAdjustedYCordAbs','arenaAdjustedXCordABS','homeTeamGoals','awayTeamGoals',
                    'location','id','shotAngle','team'], axis=1)
no_x_df = pd.read_csv('data/shots_sample.csv')
data_no_x = no_x_df.drop(['shotID','homeTeamCode','awayTeamCode','game_id','homeTeamWon','isHomeTeam','location',
                    'playerNumThatDidEvent','playerNumThatDidLastEvent','arenaAdjustedShotDistance','arenaAdjustedXCord',
                    'arenaAdjustedYCord','arenaAdjustedYCordAbs','arenaAdjustedXCordABS','xCord','yCord','homeTeamGoals','awayTeamGoals',
                    'location','id','shotAngle', 'lastEventxCord','lastEventyCord','team','goalieNameForShot','shooterLeftRight',
                    'goalieIdForShot','xFroze','xRebound','xPlayContinuedInZone','xPlayContinuedOutsideZone','xPlayStopped','xShotWasOnGoal', 'goal','xGoal'], axis=1).select_dtypes(include="number")
cord_sample = pd.read_csv('data/shots_cord_sample.csv')
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Washington_Capitals.svg/1920px-Washington_Capitals.svg.png"
rink = NHLRink(
        center_logo={
            "feature_class": CircularImage,
            "image_path": url,

            # The dimensions the image will be resized to.
            "length": 20, "width": 18,

            "x": 0, "y": 0,    # Center ice.
            "radius": 14,    # The radius of the circle for clipping.
            "zorder": 11,
        })

st.sidebar.header("Table of Contents")
option = st.sidebar.radio('Select Page', ['Introduction', 'Visualizations', 'Predictions', 'Conclusion'])

match option:
    case 'Introduction':
        st.header("Hockey Goal Attempt Analysis")
        st.image("images/greatchase.webp")
        st.markdown("""
                    ## Introduction
                    - This project creates a model that predicts the probability of a goal based off of where the player is and certain conditions on the rink
                    ### Data
                    - The data used includes every goal attempt from 2007 until 2023
                        - Includes succcesful Goals, Misses, and goal attempts that were blocked
                        - Does not include passes
                    """)
        st.subheader("Sample of Full Dataset")
        st.table(data.sample(5))


    case "Visualizations":
        
        st.header("Visualizations")
        st.markdown(
            """
            ## X/Y Coordinates
            - Visualizations and Predictions utilize only adjusted X/Y coordinates
                - All X Coordinates are positive
                - Y Coordinates from the left side of the rink are inverted
                - Makes it so every shot would appear to come from the right end of the rink
                - Allows for consistent visualizations and data manipulation
            ## Example
            """
        )
        fig, ax = plt.subplots()
        rink.scatter(ax=ax,x=-55,y=-22, color='g', s=250)
        rink.scatter(ax=ax, x=55, y=22, color='b', s=250)
        rink.text(
                [-55,55], [-22,22], ["original","adjusted"],
                color="white",
                ha="center", va="center",
                fontsize=4,
            )
        ax.set_facecolor("black")
        fig.set_facecolor("black")
        st.pyplot(fig)
        st.subheader("Example of X/Y Coordinate Adjustments inside Dataset")
        st.table(cord_sample)
        st.header("Hexbin charts")
        tab3, tab4 = st.tabs(["Teams","Players"])
        with tab3:
            st.write("Team Codes in Dataset")
            teams = ['ANA', 'ARI', 'ATL', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY',
                    'CHI', 'COL', 'DAL', 'DET', 'EDM', 'FLA', 'L.A', 'LAK', 'MIN',
                    'MTL', 'N.J', 'NJD', 'NSH', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT',
                    'S.J', 'SEA', 'SJS', 'STL', 'T.B', 'TBL', 'TOR', 'VAN', 'VGK', 'WPG', 'WSH']
            cols_per_row = 8
            for i in range(0, len(teams), cols_per_row):
                row = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(teams):
                        row[j].write(teams[i + j])
            team_selection = st.text_input("Enter a Team", value="WSH")
            st.image(f"images/{team_selection}_hexbin.png")
            st.markdown("""
                        ## General Trends-Teams
                        - Teams tend to have shots and misses from around the entire field
                        - Goals tend to come from directly in front of the net, regardless of team
                        """)
            invert_team = st.checkbox("Invert Color")
            if invert_team:
                st.image(f"images/{team_selection}_inverted_hexbin.png")
        with tab4:
            st.write("Top 32 Players By Shot Totals in Dataset")
            players = ['Alex Ovechkin', 'Brent Burns', 'Patrick Kane', 'Joe Pavelski', 'Jeff Carter', 'Sidney Crosby',
                        'Evgeni Malkin', 'Phil Kessel', 'Patrice Bergeron', 'Steven Stamkos', 'John Tavares', 'Corey Perry',
                        'Tyler Seguin', 'Zach Parise', 'Eric Staal', 'Nathan MacKinnon', 'Evander Kane', 'Anze Kopitar',
                        'Max Pacioretty', 'Claude Giroux', 'Dustin Brown', 'Erik Karlsson', 'Jamie Benn', 'Patrick Marleau',
                        'Kris Letang', 'Blake Wheeler', 'Jeff Skinner', 'Henrik Zetterberg', 'Shea Weber', 'Rick Nash', 'Brad Marchand', 'Alex Pietrangelo']
            cols_per_row = 8
            for i in range(0, len(players), cols_per_row):
                row = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(players):
                        row[j].write(players[i + j])
            first_name = st.text_input("Enter a Player's First Name", value="Alex")
            last_name = st.text_input("Enter a Player's Last Name", value="Ovechkin")
            if st.button("Load Hexbin"):
                st.image(f"images/{first_name}_{last_name}_hexbin.png")
            st.markdown("""
                        ## General Trends-Players
                        - Player Shot/Goal/Miss attempts are vastly more varied
                        - Hotspot for Goals are still near the front of the goal
                        - Some key players tendencies can be seen on the overall team charts
                            - Ovechkin's shots from the left face-off circle are visible, if faint, on the Capital's map
                        """)

    case "Predictions":
        st.header("Predicting Expected Goal")
        xcord = st.slider("X Coordinate",0,100,85)
        ycord = st.slider("Y Coordinate",-43,43,0)
        home_skaters = st.slider("Home skaters on ice (excluding goalie)",3,6,5)
        away_skaters = st.slider("Away skaters on ice (excluding goalie)",3,6,5)
        on_goal = st.selectbox("Shot on Goal",[0,1],1)
        model_selection = st.selectbox("Select A Model Type",["Overall League Model", "Alex Ovechkin Model", "Recent Models"])
        st.subheader("Randomize Predictors")
        if st.button("Run Predictors"):
            predictors_recent = random_row_except(data_no_x, ["xCordAdjusted","yCordAdjusted",
                                                "homeSkatersOnIce","awaySkatersOnIce","shotWasOnGoal"])
            predictors_recent = predictors_recent.replace(np.nan, 0)
            predictors_recent['shotGoalieFroze'] = 0
            predictors_recent['shotPlayStopped'] = 0
            predictors_recent['shotWasOnGoal'] = on_goal
            predictors_recent['xCordAdjusted'] = xcord
            predictors_recent['yCordAdjusted'] = ycord
            predictors_recent['homeSkatersOnIce'] = home_skaters
            predictors_recent['awaySkatersOnIce'] = away_skaters
        else:
            predictors_recent = data_no_x.sample(1).replace(np.nan, 0)
            predictors_recent['shotGoalieFroze'] = 0
            predictors_recent['shotPlayStopped'] = 0
            predictors_recent['shotWasOnGoal'] = on_goal
            predictors_recent['xCordAdjusted'] = xcord
            predictors_recent['yCordAdjusted'] = ycord
            predictors_recent['homeSkatersOnIce'] = home_skaters
            predictors_recent['awaySkatersOnIce'] = away_skaters

        match model_selection:
            case "Overall League Model":
                model = joblib.load("hockey_regression_model.pkl")
                predictors = pd.DataFrame({'timeUntilNextEvent': [0],
                                        'shotGoalieFroze':[0],
                                        'shotPlayContinuedInZone':[0],
                                        'shotGeneratedRebound':[0],
                                        'shotPlayStopped':[0],
                                        'shotWasOnGoal':[on_goal],
                                        'xCordAdjusted':[xcord],
                                        'yCordAdjusted':[ycord],
                                        'homeSkatersOnIce':[home_skaters],
                                        'awaySkatersOnIce':[away_skaters]})
                y_pred = model.predict(predictors)
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Expected Chance of Goal")
                    st.header(f":blue-background[{y_pred[0]:.2f}]")
                with col2:
                    st.header("Location of Shot")
                    fig, ax = plt.subplots(figsize=(8,4))
                    rink.draw(ax=ax, display_range='offense')
                    rink.scatter(xcord, ycord, color="g", s=150)
                    st.pyplot(fig, use_container_width=False)

            case "Alex Ovechkin Model":
                model = joblib.load("ovi_model_regression.pkl")
                predictors = pd.DataFrame({'timeUntilNextEvent': [0],
                                        'shotGoalieFroze':[0],
                                        'shotPlayContinuedInZone':[0],
                                        'shotGeneratedRebound':[0],
                                        'shotPlayStopped':[0],
                                        'shotWasOnGoal':[on_goal],
                                        'xCordAdjusted':[xcord],
                                        'yCordAdjusted':[ycord],
                                        'homeSkatersOnIce':[home_skaters],
                                        'awaySkatersOnIce':[away_skaters]})
                y_pred = model.predict(predictors)
                col3, col4 = st.columns(2)
                with col3:
                    st.header("Expected Chance of Goal")
                    st.header(f":blue-background[{y_pred[0]:.2f}]")
                with col4:
                    st.header("Location of Shot")
                    fig, ax = plt.subplots(figsize=(8,4))
                    rink.draw(ax=ax, display_range='offense')
                    rink.scatter(xcord, ycord, color="g", s=150)
                    st.pyplot(fig, use_container_width=False)
            case "Recent Models":
                model_class = joblib.load("recent_classifier_model.pkl")
                y_pred_class = model_class.predict_proba(predictors_recent)
                y_pred_class_true = model_class.predict(predictors_recent)
                col5, col6 = st.columns(2)
                with col5:
                    st.header("Expected Chance of Goal")
                    st.markdown("""
                                ## Predicts Yes/No for Goal
                                - Weighs probability of Goal and probability No Goal
                                - First return is the probability of Goal
                                - Second return is model's prediction
                                """)
                    st.header(f"Probability of Goal: :blue-background[{y_pred_class[0][1]:.2f}]")
                    if y_pred_class_true[0] == 1:
                        st.header("Predicted Outcome: :blue-background[Goal!]")
                    else:
                        st.header("Predicted Outcome: :blue-background[No Goal]")
                with col6:
                    st.header("Location of Shot")
                    fig, ax = plt.subplots(figsize=(8,4))
                    rink.draw(ax=ax, display_range='offense')
                    rink.scatter(xcord, ycord, color="g", s=150)
                    st.pyplot(fig, use_container_width=False)
        

    case "Conclusion":
        st.markdown("""
                    ## Conclusions
                    - Teams generally score in the area directly in front of the goal, with some variation
                    - Players have a greater spread of where they shoot and score from
                    ## Further Research
                    - Win probability prediction
                        - Stanley Cup playoff winner prediction models
                    - Team Specific data
                    - Player Specific data
                    ## Contact Info
                    - Email: russell.h.bonett.mil@army.mil
                    - GitHub: https://github.com/RussBone/ddi-final-project
                    ## Resources
                    - MoneyPuck.com
                        - https://moneypuck.com/data.htm
                    """)