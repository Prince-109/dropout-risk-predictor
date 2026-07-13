<div align="center">

# 🎓 EduGuard — AI-Based Dropout Prediction & Counseling System

**Predicting student dropout risk early — so institutions can act before it's too late.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

*Built for Smart India Hackathon — Problem Statement SIH25102*

</div>

---

## 🎯 What It Does

EduGuard analyzes student data — attendance, marks, fee payment history, and more — to **predict dropout risk before it happens**, giving institutions the lead time to intervene through counseling, alerts, and targeted support.

## 🚀 Features

- 📊 **Interactive Streamlit dashboard** for real-time risk visualization
- 🧠 **ML-based dropout risk prediction** trained on student behavioral data
- 📈 **Model training & evaluation** built into the pipeline
- 📬 **Notification-ready** — Email / SMS alert infrastructure (Twilio-compatible)
- ⚙️ **SMTP port checking utility** for reliable email delivery
- 🔐 **Secure environment variable handling** for credentials

## 🧠 How It Works

1. Load student dataset (attendance, marks, fees, etc.)
2. Preprocess and clean the data
3. Train a machine learning model on historical outcomes
4. Save the trained model as `risk_model.pkl`
5. Load the model inside the Streamlit app
6. Predict dropout risk per student
7. Surface insights and risk levels through the dashboard

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| App Framework | Streamlit |
| ML | scikit-learn, Joblib |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Config / Secrets | python-dotenv |
| Notifications | Twilio (optional, SMS) |

## 📁 Project Structure

```
eduguard/
├── app.py                     # Main Streamlit application
├── train_model.py             # Model training script
├── risk_model.pkl             # Trained ML dropout risk model
├── check_smtp_ports.py        # SMTP port availability checker
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Prince-109/eduguard.git
cd eduguard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
# Create a .env file with:
# EMAIL_USER=your_email@example.com
# EMAIL_PASSWORD=your_email_password
# TWILIO_SID=your_twilio_sid
# TWILIO_AUTH_TOKEN=your_twilio_auth_token

# 4. Run the app
streamlit run app.py
```

## 🖥️ Usage

1. Launch the Streamlit app
2. Load or upload student data
3. View analytics and risk statistics
4. Run dropout risk prediction
5. Identify high-risk students
6. Use insights for counseling and early intervention

## 📸 Screenshots / Demo

<!-- Add a screenshot of the dashboard showing risk levels — this is the single highest-impact addition here. -->

## 🔮 Future Enhancements

- [ ] Improve accuracy with deep learning models
- [ ] Real-time student monitoring
- [ ] Role-based dashboards (Admin / Counselor)
- [ ] Cloud deployment (AWS / Render / Hugging Face)
- [ ] Mobile and SMS alert integration

## 👥 Team

Built for Smart India Hackathon — Problem Statement SIH25102.

**Prince Kumar** — [LinkedIn](https://linkedin.com/in/prince-kumar-493037195) · [GitHub](https://github.com/Prince-109)
