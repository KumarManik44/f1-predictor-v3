import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="🏎️ F1 Predictor V3",
    page_icon="🏁",
    layout="wide"
)


# Load model
@st.cache_resource
def load_model():
    with open('models/ensemble/cat_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


# Load historical data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/f1_v3_complete_features.csv')
    return df


# Feature columns (47 features)
FEATURE_COLUMNS = [
    'grid_position', 'front_row_start', 'quali_made_q3', 'quali_made_q2',
    'quali_best_time', 'quali_gap_to_pole', 'quali_gap_to_pole_pct',
    'quali_performance_score', 'quali_q1_q2_improvement', 'quali_q2_q3_improvement',
    'driver_last3_avg_points', 'driver_last3_avg_position',
    'driver_last5_avg_points', 'driver_last5_avg_position',
    'driver_season_points', 'driver_season_races', 'driver_last5_podiums',
    'driver_dnf_rate', 'driver_avg_finish_position', 'driver_championship_position',
    'constructor_last3_avg_points', 'constructor_last5_avg_points',
    'constructor_season_points', 'constructor_championship_position',
    'constructor_dnf_rate', 'constructor_avg_quali_position',
    'constructor_points_per_race', 'constructor_is_top_team',
    'circuit_driver_wins', 'circuit_driver_podiums', 'circuit_driver_avg_finish',
    'circuit_driver_experience', 'circuit_constructor_wins', 'circuit_constructor_podiums',
    'circuit_driver_best_grid', 'circuit_driver_win_rate', 'circuit_driver_podium_rate',
    'circuit_driver_points_per_race', 'circuit_avg_position_change',
    'driver_momentum', 'points_gap_to_leader', 'must_win_pressure',
    'teammate_gap', 'driver_consistency_score', 'avg_quali_race_delta',
    'season_progress', 'driver_career_races'
]

# 2025 Driver list
DRIVERS = [
    'VER', 'TSU', 'NOR', 'PIA', 'LEC', 'HAM', 'RUS', 'ANT',
    'ALO', 'STR', 'GAS', 'COL', 'LAW', 'HAD', 'BEA', 'OCO',
    'ALB', 'SAI', 'HUL', 'BOR'
]

# Circuit-specific adjustments (based on historical overtaking data)
CIRCUIT_CHARACTERISTICS = {
    "🇧🇷 Brazil GP (Interlagos)": {
        "overtaking_factor": 0.8,  # High overtaking (wet weather, elevation changes)
        "rain_probability": 0.6,
        "grid_importance": 0.6  # REDUCED from 1.0
    },
    "🇺🇸 Las Vegas GP": {
        "overtaking_factor": 0.7,
        "rain_probability": 0.0,
        "grid_importance": 0.8
    },
    "🇶🇦 Qatar GP (Losail)": {
        "overtaking_factor": 0.4,  # Low overtaking
        "rain_probability": 0.0,
        "grid_importance": 0.9
    },
    "🇦🇪 Abu Dhabi GP (Yas Marina)": {
        "overtaking_factor": 0.3,  # Very low overtaking
        "rain_probability": 0.0,
        "grid_importance": 0.95
    }
}

REMAINING_RACES = list(CIRCUIT_CHARACTERISTICS.keys())

# Header
st.title("🏎️ F1 Race Predictor V3 (Improved)")
st.markdown("**AI-Powered Podium Prediction with Circuit Characteristics**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Model Stats")
    st.metric("Base Accuracy", "93.89%")
    st.metric("Features", "47")
    st.metric("Model", "CatBoost")

    st.markdown("---")

    # Championship Standings
    st.markdown("### 🏆 2025 Championship")
    st.markdown("*After Mexico GP (R20/24)*")

    standings_data = {
        'Driver': ['NOR', 'PIA', 'VER', 'RUS', 'LEC'],
        'Points': [357, 356, 321, 258, 210]
    }
    standings_df = pd.DataFrame(standings_data)

    for idx, row in standings_df.iterrows():
        if idx == 0:
            st.markdown(f"**🥇 {row['Driver']} - {row['Points']}** 🔥")
        elif idx == 1:
            st.markdown(f"**🥈 {row['Driver']} - {row['Points']}** (-1)")
        else:
            gap = standings_df.iloc[0]['Points'] - row['Points']
            st.markdown(f"{idx + 1}. {row['Driver']} - {row['Points']} (-{gap})")

    st.markdown("---")
    st.markdown("### ⚙️ Circuit Adjustments")
    st.markdown("""
    • Interlagos: High overtaking
    • Las Vegas: Medium overtaking
    • Qatar: Low overtaking
    • Abu Dhabi: Very low overtaking
    """)

# Main app
st.header("🏁 Quick Prediction")
st.markdown("*Circuit-aware predictions with overtaking probability*")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📋 Qualifying Grid (Top 10)")

    subcol1, subcol2 = st.columns(2)

    quali_grid = {}
    for i in range(1, 11):
        col = subcol1 if i <= 5 else subcol2
        with col:
            driver = st.selectbox(
                f"P{i}",
                options=DRIVERS,
                key=f"p{i}",
                index=min(i - 1, len(DRIVERS) - 1)
            )
            quali_grid[i] = driver

with col2:
    st.subheader("🏁 Race Selection")

    selected_race = st.selectbox(
        "Select Race",
        options=REMAINING_RACES,
        index=0
    )

    # Get circuit characteristics
    circuit_char = CIRCUIT_CHARACTERISTICS[selected_race]

    # Display circuit info
    st.info(f"**Overtaking Factor:** {circuit_char['overtaking_factor']:.0%}\n\n"
            f"**Grid Importance:** {circuit_char['grid_importance']:.0%}")

    st.markdown("---")

    predict_button = st.button("🚀 Predict Podium", type="primary", use_container_width=True)

# Prediction
if predict_button:
    with st.spinner("🤖 Running circuit-aware prediction..."):
        try:
            model = load_model()
            df_hist = load_data()

            results = []

            for pos in range(1, 11):
                driver = quali_grid[pos]
                driver_data = df_hist[df_hist['driverCode'] == driver]

                if len(driver_data) > 0:
                    features = {}

                    # ADJUSTED: Reduce grid position impact based on circuit
                    grid_penalty = (pos - 1) * (1 - circuit_char['overtaking_factor'])
                    adjusted_grid = 1 + grid_penalty

                    features['grid_position'] = adjusted_grid  # CIRCUIT-ADJUSTED!
                    features['front_row_start'] = 1 if pos <= 2 else 0
                    features['quali_made_q3'] = 1 if pos <= 10 else 0
                    features['quali_made_q2'] = 1 if pos <= 15 else 0

                    recent = driver_data.tail(1).iloc[0]

                    for col in FEATURE_COLUMNS:
                        if col not in features:
                            if col in recent.index:
                                features[col] = recent[col]
                            else:
                                features[col] = 0

                    # Boost driver form for high-overtaking circuits
                    if circuit_char['overtaking_factor'] > 0.6:
                        features['driver_last5_avg_points'] *= 1.3
                        features['circuit_driver_win_rate'] *= 1.5

                    X = pd.DataFrame([features])[FEATURE_COLUMNS].fillna(0)
                    prob = model.predict_proba(X)[0][1]

                    # Apply circuit-specific adjustment to probability
                    if pos > 3:  # Boost lower grid positions on overtaking circuits
                        prob *= (1 + circuit_char['overtaking_factor'] * 0.3)

                    results.append({
                        'Position': pos,
                        'Driver': driver,
                        'Actual_Grid': pos,
                        'Adjusted_Grid': adjusted_grid,
                        'Podium_Probability': prob
                    })
                else:
                    results.append({
                        'Position': pos,
                        'Driver': driver,
                        'Actual_Grid': pos,
                        'Adjusted_Grid': pos,
                        'Podium_Probability': 0.15 if pos <= 10 else 0.05
                    })

            results_df = pd.DataFrame(results).sort_values('Podium_Probability', ascending=False)

            # Display results
            st.markdown("---")
            st.header(f"🏆 Predicted Podium: {selected_race}")

            # Show circuit adjustment impact
            if circuit_char['overtaking_factor'] > 0.6:
                st.success(
                    f"⚡ High overtaking circuit! Lower grid positions boosted by {circuit_char['overtaking_factor'] * 30:.0f}%")

            # Top 3
            col1, col2, col3 = st.columns(3)
            top3 = results_df.head(3)

            with col1:
                st.markdown("### 🥇 1st Place")
                st.markdown(f"## **{top3.iloc[0]['Driver']}**")
                st.metric("Started", f"P{int(top3.iloc[0]['Actual_Grid'])}")
                st.metric("Probability", f"{top3.iloc[0]['Podium_Probability']:.1%}")

            with col2:
                st.markdown("### 🥈 2nd Place")
                st.markdown(f"## **{top3.iloc[1]['Driver']}**")
                st.metric("Started", f"P{int(top3.iloc[1]['Actual_Grid'])}")
                st.metric("Probability", f"{top3.iloc[1]['Podium_Probability']:.1%}")

            with col3:
                st.markdown("### 🥉 3rd Place")
                st.markdown(f"## **{top3.iloc[2]['Driver']}**")
                st.metric("Started", f"P{int(top3.iloc[2]['Actual_Grid'])}")
                st.metric("Probability", f"{top3.iloc[2]['Podium_Probability']:.1%}")

            # Full results
            st.markdown("---")
            st.subheader("📊 Full Top 10 Predictions")

            display_df = results_df[['Driver', 'Actual_Grid', 'Podium_Probability']].copy()
            display_df['Podium_Probability'] = display_df['Podium_Probability'].apply(lambda x: f"{x:.1%}")
            display_df['Actual_Grid'] = display_df['Actual_Grid'].apply(lambda x: f"P{int(x)}")
            display_df.columns = ['Driver', 'Grid', 'Probability']

            st.dataframe(display_df, hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"❌ {str(e)}")

st.markdown("---")
st.markdown("*F1 Predictor V3 | Circuit-Aware Predictions | Oct 2025*")
