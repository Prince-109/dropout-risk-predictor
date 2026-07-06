# 🎓 AI-Based Dropout Prediction and Counseling System

**SIH25102 Project**

This project is an **AI-based student dropout prediction and counseling system** built using **Machine Learning and Streamlit**.  
It analyzes student-related data to predict dropout risk and helps institutions take early preventive actions through insights and notifications.

---

## 🚀 Project Features

- 📊 Interactive Streamlit dashboard
- 🧠 Machine Learning–based dropout risk prediction
- 📈 Model training and evaluation
- 📬 Notification readiness (Email / SMS)
- ⚙️ SMTP port checking utility
- 🔐 Secure environment variable usage

---

## 📂 Project Structure

```text
ai-based-dropout-prediction/
│
├── app.py                     # Main Streamlit application
├── train_model.py             # Model training script
├── risk_model.pkl             # Trained ML dropout risk model
├── check_smtp_ports.py        # SMTP port availability checker
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```
## 🧠 Machine Learning Workflow

1. Load student dataset (attendance, marks, fees, etc.)
2. Preprocess and clean the data
3. Train a machine learning model
4. Save the trained model as `risk_model.pkl`
5. Load the model inside the Streamlit application
6. Predict student dropout risk
7. Display insights and risk levels through the dashboard

## ⚙️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- python-dotenv
- Twilio (optional for SMS alerts)

---

## 🖥️ Application Usage

1. Launch the Streamlit application
2. Load or upload student data
3. View analytics and statistics
4. Run dropout risk prediction
5. Identify high-risk students
6. Use insights for counseling and intervention

## 📦 Model File

- **risk_model.pkl**  
  Contains the trained machine learning model used to predict student dropout risk.  
  This file is automatically loaded by the Streamlit application during runtime.

---

## 🔐 Environment Variables

Create a `.env` file to store sensitive credentials:

```env
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_email_password
TWILIO_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token

```
## 🚀 Future Enhancements

- Improve accuracy using deep learning models
- Add real-time student monitoring
- Implement role-based dashboards (Admin / Counselor)
- Deploy the application on cloud platforms (AWS / Render / Hugging Face)
- Enable mobile and SMS alert integration

## 👨‍💻 Author

**Ashok Sankhla**  
Computer Science Engineering (AI)  
Machine Learning & Data Science Enthusiast

