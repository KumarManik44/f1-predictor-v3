# 🏎️ F1 Race Predictor V3

**AI-Powered Formula 1 Race Outcome Prediction using Machine Learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow.svg)](https://catboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🚀 Live Demo

[**Try the App Here**](https://f1-predictor-v3.streamlit.app/)

## 📊 Project Overview

F1 Race Predictor V3 is a machine learning system that predicts Formula 1 race podium finishers with **93.89% accuracy** on test data. The model analyzes qualifying results, driver form, constructor performance, and circuit-specific factors to forecast race outcomes.

### ✨ Key Features

- **93.89% Test Accuracy** - Validated on 2025 R11-R19 races
- **47 Advanced Features** - Qualifying intelligence, driver momentum, constructor strength
- **Real-time Predictions** - Interactive Streamlit web app
- **Historical Analysis** - 2022-2025 season data (1,738 race results)
- **Production Ready** - Deployed model with current 2025 championship standings

### 🎯 Model Performance

| Metric | V2 (Baseline) | V3 (Current) | Improvement |
|--------|--------------|--------------|-------------|
| Test Accuracy | 91.29% | **93.89%** | **+2.60%** |
| Features | 36 | 47 | +11 |
| Training Data | 1,500 records | 1,738 records | +16% |
| Podium Precision | 79% | 83% | +4% |
| Podium Recall | 65% | 70% | +5% |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- 4GB RAM minimum

- ### Installation

1. **Clone the repository**
git clone https://github.com/yourusername/f1-predictor-v3.git
cd f1-predictor-v3

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the Streamlit app**
streamlit run app_v3.py

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
f1-predictor-v3/
│
├── app_v3.py # Main Streamlit application
├── requirements.txt 
│
├── notebooks/
│ └── f1_predictor_v3_master.ipynb
│
├── data/
│ ├── raw/
│ │ ├── race_results_2022_2025.csv
│ │ └── qualifying_results_2022_2025.csv
│ └── processed/
│ └── f1_v3_complete_features.csv
│
├── models/
│ └── ensemble/
│ └── cat_model.pkl # Trained CatBoost model
│
├── mlflow/ # MLflow experiment tracking
│
└── README.md
```


---

## 🔬 Methodology

### Data Collection

**Sources:**
- [Ergast API](http://ergast.com/mrd/) - Historical race results (2022-2025)
- [FastF1](https://docs.fastf1.dev/) - Qualifying lap times and telemetry

**Dataset:**
- **1,738 race results** (87 races × ~20 drivers)
- **1,759 qualifying sessions** (88 sessions including Mexico GP test data)
- **2022-2024 + 2025 R1-R19** for training
- **2025 R11-R19** held out for testing

### Feature Engineering (47 Features)

#### 1. Qualifying Intelligence (10 features)
- Grid position, front row start indicator
- Q1/Q2/Q3 lap times and improvements
- Gap to pole (absolute & percentage)
- Qualifying performance score (normalized 0-1)

#### 2. Driver Performance (10 features)
- Last 3/5 races average points and position
- Season cumulative points and races
- Podium count in last 5 races
- DNF rate, career average finish
- Championship position

#### 3. Constructor Performance (8 features)
- Team form (last 3/5 races points)
- Constructor championship position
- Reliability (DNF rate)
- Average qualifying position
- Top team indicator (top 3 in standings)

#### 4. Circuit-Specific (11 features)
- Driver wins/podiums at circuit
- Average finish position at circuit
- Circuit experience (total races)
- Constructor performance at circuit
- Best historical grid position
- Win rate and podium rate at circuit

#### 5. Advanced Metrics (8 features)
- Driver momentum (form trend)
- Points gap to championship leader
- Must-win pressure indicator
- Teammate gap
- Consistency score
- Qualifying vs race performance delta
- Season progress percentage
- Career race count

### Model Architecture

**Algorithm:** CatBoost Classifier
- **n_estimators:** 150
- **max_depth:** 6
- **learning_rate:** 0.05
- **Regularization:** L1=0.1, L2=1.0
- **Sampling:** 80% subsample, 80% column sample

**Training Strategy:**
- Train on: 2022-2024 + 2025 R1-R10 (1,558 records)
- Test on: 2025 R11-R19 (180 records)
- 5-fold stratified cross-validation
- Class-weighted for imbalanced data (15% podium, 85% non-podium)

---

## 📈 Results & Validation

### Test Set Performance (2025 R11-R19)

Accuracy: 93.89%
Precision: 82.61% (podium predictions)
Recall: 70.37% (actual podiums caught)
F1-Score: 76.00%


**Classification Report:**
             precision    recall  f1-score   support   accuracy
Non-Podium   0.95         0.97    0.96       153       0.93
Podium       0.83         0.70    0.76       27        180



### Real-World Validation: Mexico GP 2025

**Predicted Top 3:**
1. Lando Norris (NOR) ✅
2. Charles Leclerc (LEC) ✅  
3. George Russell (RUS) ❌

**Actual Top 3:**
1. Lando Norris (NOR)
2. Charles Leclerc (LEC)
3. Max Verstappen (VER)

**Accuracy:** 2/3 podium finishers (66.7%)

**Analysis:** Model correctly predicted top 2 but underestimated Verstappen's race pace from P5 grid position.

---

## 🎮 Usage

### Web Application

1. **Select qualifying grid** - Choose drivers P1-P10
2. **Select race** - Brazil GP, Las Vegas, Qatar, or Abu Dhabi
3. **Predict** - Model generates podium probability for each driver
4. **View results** - Top 3 predicted finishers with confidence scores


---

## 🔑 Key Insights

### Feature Importance

| Feature | Importance | Insight |
|---------|-----------|---------|
| Grid Position | 16.0% | Qualifying performance is strongest predictor |
| Constructor Form | 7.9% | Team strength matters significantly |
| Qualifying Gap % | 5.7% | Relative pace indicator |
| Driver Last 5 Points | 4.8% | Recent form is crucial |
| Front Row Start | 4.8% | P1/P2 grid advantage |

### Model Strengths

✅ **Normal races** - 93.89% accurate for standard race conditions  
✅ **Qualifying analysis** - Strong integration of Q1/Q2/Q3 data  
✅ **Constructor impact** - Successfully models team performance  
✅ **Momentum tracking** - Captures driver form trends  

### Known Limitations

⚠️ **Grid position bias** - Over-relies on starting position (16% importance)  
⚠️ **Circuit specificity** - Uses career stats, not actual circuit-specific data  
⚠️ **Weather integration** - No real-time weather data (planned for V4)  
⚠️ **Race strategy** - Doesn't model pit stops, tire choice, safety cars  
⚠️ **Chaos races** - Underperforms on unpredictable races (e.g., Brazil 2024: Verstappen P17→P1)  

---

## 🛠️ Technical Stack

- **Python 3.8+** - Core language
- **FastF1** - F1 telemetry and timing data
- **Ergast API** - Historical race results
- **CatBoost** - Gradient boosting model
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **Scikit-learn** - ML utilities and metrics
- **MLflow** - Experiment tracking

---

## 🗺️ Roadmap to V4

### Planned Improvements

1. **True Circuit-Specific Features**
   - Per-circuit historical performance
   - Track-specific overtaking success rates
   - Circuit mastery indicators

2. **Weather Integration**
   - OpenWeather API live data
   - Historical weather patterns
   - Wet weather driver skill ratings

3. **Race Strategy Features**
   - Pit stop timing prediction
   - Tire degradation modeling
   - Safety car probability

4. **Two-Stage Model**
   - Stage 1: Predict race type (normal/chaos)
   - Stage 2A: Normal race model (grid-heavy)
   - Stage 2B: Chaos race model (skill-heavy)

---

## 📊 Model Comparison

### V1 → V2 → V3 Evolution

| Version | Accuracy | Features | Key Innovation |
|---------|----------|----------|----------------|
| V1 | 87.5% | 20 | Basic statistics |
| V2 | 91.29% | 36 | Rolling averages + MLflow |
| **V3** | **93.89%** | **47** | **Qualifying intelligence** |
| V4 (Planned) | Target: 96%+ | 60+ | Weather + strategy + circuit mastery |

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add real weather API integration
- [ ] Implement circuit-specific historical features
- [ ] Build pit stop strategy predictor
- [ ] Create race simulation module
- [ ] Add driver wet weather skill ratings

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👏 Acknowledgments

- **Ergast API** - Historical F1 data
- **FastF1** - Telemetry and timing data
- **F1 Community** - Testing and feedback
- **CatBoost Team** - Excellent gradient boosting library

---

## ⭐ Star This Project

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Built with ❤️ for Formula 1 fans and data science enthusiasts**

*Last Updated: October 29, 2025 - After Mexico GP (Round 20/24)*
