# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
import joblib
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client

load_dotenv()

MODEL_PATH = 'models/risk_model.pkl'
SAMPLE_DIR = 'data'

st.set_page_config(page_title='Dropout Prediction Dashboard', layout='wide', page_icon='🎓')

# Custom CSS for classy look
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
* {
    font-family: 'Poppins', sans-serif;
}
body {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    color: #333;
}
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
.main-header h1 {
    font-size: 3em;
    margin: 0;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.main-header p {
    font-size: 1.3em;
    margin: 15px 0 0 0;
    opacity: 0.9;
}
.sidebar .sidebar-content {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
}
.card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    margin-bottom: 25px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.15);
}
.metric-card {
    text-align: center;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    padding: 20px;
    margin: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: scale(1.05);
}
.metric-value {
    font-size: 2.5em;
    font-weight: bold;
    color: #667eea;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.metric-label {
    font-size: 1em;
    color: #666;
    font-weight: 500;
}
.footer {
    text-align: center;
    padding: 25px;
    background: linear-gradient(135deg, #343a40 0%, #495057 100%);
    color: white;
    border-radius: 15px;
    margin-top: 30px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.stPlotlyChart {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 AI-based Dropout Prediction & Counseling</h1>
    <p>Smart India Hackathon 2025 — SIH25102</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Empowering Education with AI-Driven Insights")

# --------- Page Functions ---------

def show_dashboard():
    """Main Dashboard Page"""
    st.header('📊 Main Dashboard')

    # Data loading and processing (same as before)
    if df.empty:
        st.warning('Data missing — sample will be used if available.')
        return

    if st.button('Prepare & Train Model (quick)', key='dashboard_train'):
        processed_df = preprocess(df)
        if not processed_df.empty:
            processed_df = create_rule_labels(processed_df)
            clf = train_and_save_model(processed_df)
            if clf:
                st.success('Model trained and saved to models/risk_model.pkl')
            else:
                st.error("Failed to train model.")
        else:
            st.error("Preprocessing failed.")

    model = load_model()

    if st.button('Run Prediction', key='dashboard_predict'):
        if model is None:
            st.error('Model not found. Pehle "Prepare & Train Model" karlo.')
        else:
            processed_df = preprocess(df)
            if not processed_df.empty:
                processed_df = predict_with_model(model, processed_df)
                processed_df = apply_rule_engine(processed_df)

                # Store processed_df in session state for other pages
                st.session_state.processed_df = processed_df

                # Metrics Cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="metric-card"><div class="metric-value">' + str(len(processed_df)) + '</div><div class="metric-label">Total Students</div></div>', unsafe_allow_html=True)
                with col2:
                    high_count = len(processed_df[processed_df['final_risk'] == 'High'])
                    st.markdown('<div class="metric-card"><div class="metric-value">' + str(high_count) + '</div><div class="metric-label">High Risk</div></div>', unsafe_allow_html=True)
                with col3:
                    medium_count = len(processed_df[processed_df['final_risk'] == 'Medium'])
                    st.markdown('<div class="metric-card"><div class="metric-value">' + str(medium_count) + '</div><div class="metric-label">Medium Risk</div></div>', unsafe_allow_html=True)
                with col4:
                    low_count = len(processed_df[processed_df['final_risk'] == 'Low'])
                    st.markdown('<div class="metric-card"><div class="metric-value">' + str(low_count) + '</div><div class="metric-label">Low Risk</div></div>', unsafe_allow_html=True)

                # Interactive Gauge Charts
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader('📊 Key Performance Indicators')

                # Calculate overall metrics
                avg_attendance = processed_df['attendance_percent'].mean()
                avg_score = processed_df['current_avg'].mean()

                col1, col2 = st.columns(2)
                with col1:
                    # Attendance Gauge
                    fig_attendance = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_attendance,
                        title={'text': "Average Attendance %"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "red"},
                                {'range': [50, 75], 'color': "yellow"},
                                {'range': [75, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 75
                            }
                        }
                    ))
                    st.plotly_chart(fig_attendance, use_container_width=True)

                with col2:
                    # Average Score Gauge
                    fig_score = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_score,
                        title={'text': "Average Academic Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 35], 'color': "red"},
                                {'range': [35, 60], 'color': "yellow"},
                                {'range': [60, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 60
                            }
                        }
                    ))
                    st.plotly_chart(fig_score, use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Prediction Results
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader('📊 Prediction Results')
                display_df = processed_df[
                    ['student_id', 'name', 'attendance_percent', 'current_avg',
                     'late_payments', 'predicted_risk', 'rule_risk',
                     'final_risk', 'guardian_email', 'guardian_phone']
                ].sort_values(by='final_risk', ascending=False)
                page_size = st.slider('Rows per page', min_value=10, max_value=100, value=20, step=10)
                st.dataframe(display_df.head(page_size))
                st.markdown('</div>', unsafe_allow_html=True)

                # Risk Distribution Chart
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader('📈 Risk Distribution')
                counts = processed_df['final_risk'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
                fig = px.bar(x=counts.index, y=counts.values, labels={'x': 'Risk Level', 'y': 'Count'},
                            title='Students per Risk Level', color=counts.index,
                            color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'})
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Risk Indicator
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader('🎯 Risk Indicator')
                def risk_color(risk):
                    if risk == 'High':
                        return '🔴 High Risk'
                    elif risk == 'Medium':
                        return '🟡 Medium Risk'
                    else:
                        return '🟢 Low Risk'

                processed_df['risk_indicator'] = processed_df['final_risk'].apply(risk_color)
                risk_display_df = processed_df[['name', 'risk_indicator']].sort_values(by='risk_indicator', ascending=False)
                st.dataframe(risk_display_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Email Alerts
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader('📧 Send Email Alerts')
                st.info('📝 To send emails, set EMAIL_USER and EMAIL_PASS in .env file. For Gmail, use app password (not regular password). Enable 2FA and generate app password at https://myaccount.google.com/apppasswords')
                smtp_server = st.text_input('SMTP Server', value='smtp.gmail.com', placeholder='smtp.gmail.com')
                smtp_port = st.number_input('SMTP Port', value=465, min_value=1, max_value=65535)
                test_email = st.text_input('Test Email Address', placeholder='your.email@example.com')
                if st.button('📤 Send Test Email', key='test_email'):
                    if test_email:
                        subject = "Test Email from Dropout Prediction System"
                        body = "This is a test email to verify email configuration."
                        if send_email(test_email, subject, body, smtp_server, smtp_port):
                            st.success('✅ Test email sent successfully!')
                        else:
                            st.error('❌ Failed to send test email. Check .env, SMTP settings, and credentials.')
                    else:
                        st.warning('Please enter a test email address.')
                high_df = processed_df[processed_df['final_risk'] == 'High']
                st.write(f'🔴 High risk students: {len(high_df)}')
                missing_emails = high_df['guardian_email'].isna() | (high_df['guardian_email'] == '')
                missing_count = missing_emails.sum()
                if missing_count > 0:
                    st.warning(f'⚠️ {missing_count} students have missing guardian emails and will be skipped.')
                if len(high_df) > 0:
                    success = 0
                    if st.button('🚀 Send Emails to Guardians'):
                        for _, row in high_df.iterrows():
                            to = row.get('guardian_email', '')
                            if pd.isna(to) or to == '':
                                continue
                            subject = f"Academic Performance Alert: {row['name']} - Student at Risk of Dropout"
                            body = (
                                f"Dear Guardian,\n\n"
                                f"Your ward {row['name']} (ID: {row['student_id']}) has been identified as HIGH RISK for dropout.\n\n"
                                f"Details:\n"
                                f"- Attendance: {row['attendance_percent']:.1f}%\n"
                                f"- Average Score: {row['current_avg']:.1f}\n"
                                f"- Late Payments: {int(row['late_payments'])}\n\n"
                                f"Please contact the school counselor immediately for support and guidance.\n\n"
                                f"Best regards,\n"
                                f"School Administration\n"
                                f"AI Dropout Prediction System"
                            )
                            try:
                                if send_email(to, subject, body, smtp_server, smtp_port):
                                    success += 1
                            except Exception as e:
                                st.error(f"Failed to send email to {to}: {e}")
                        st.success(f'✅ Emails sent: {success}/{len(high_df) - missing_count} (skipped {missing_count} with missing emails)')
                st.markdown('</div>', unsafe_allow_html=True)

                # WhatsApp Alerts
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader('📱 Send WhatsApp Alerts')
                st.info('📝 To send WhatsApp messages, set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_NUMBER in .env file.')
                test_phone = st.text_input('Test Phone Number (with country code, e.g. +1234567890)', placeholder='+1234567890')
                test_message = st.text_area('Test Message', value='This is a test WhatsApp message from Dropout Prediction System.')
                if st.button('📤 Send Test WhatsApp Message', key='test_whatsapp'):
                    if test_phone and test_message:
                        if send_whatsapp_message(test_phone, test_message):
                            st.success('✅ Test WhatsApp message sent successfully!')
                        else:
                            st.error('❌ Failed to send WhatsApp message. Check Twilio credentials and phone number.')
                    else:
                        st.warning('Please enter both phone number and message.')
                if len(high_df) > 0:
                    success_whatsapp = 0
                    if st.button('🚀 Send WhatsApp Alerts to Guardians'):
                        for _, row in high_df.iterrows():
                            to = row.get('guardian_phone', '')  # Use guardian phone number if available
                            if pd.isna(to) or to == '':
                                continue
                            message = (
                                f"Dear Guardian,\n\n"
                                f"Your ward {row['name']} (ID: {row['student_id']}) has been identified as HIGH RISK for dropout.\n\n"
                                f"Details:\n"
                                f"- Attendance: {row['attendance_percent']:.1f}%\n"
                                f"- Average Score: {row['current_avg']:.1f}\n"
                                f"- Late Payments: {int(row['late_payments'])}\n\n"
                                f"Please contact the school counselor immediately for support and guidance.\n\n"
                                f"Best regards,\n"
                                f"School Administration\n"
                                f"AI Dropout Prediction System"
                            )
                            try:
                                if send_whatsapp_message(to, message):
                                    success_whatsapp += 1
                            except Exception as e:
                                st.error(f"Failed to send WhatsApp message to {to}: {e}")
                        st.success(f'✅ WhatsApp messages sent: {success_whatsapp}/{len(high_df)}')
                st.markdown('</div>', unsafe_allow_html=True)

def show_student_profiles():
    """Student Profiles Page"""
    st.header('👤 Student Profiles')

    if 'processed_df' not in st.session_state or st.session_state.processed_df.empty:
        st.warning('Please run prediction on the Dashboard first to view student profiles.')
        return

    processed_df = st.session_state.processed_df

    # Student selection
    student_names = processed_df['name'].tolist()
    selected_student = st.selectbox('Select Student', student_names)

    if selected_student:
        student_data = processed_df[processed_df['name'] == selected_student].iloc[0]

        # Student Overview Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Student ID', str(student_data['student_id']))
        with col2:
            st.metric('Attendance %', f"{student_data['attendance_percent']:.1f}%")
        with col3:
            st.metric('Current Avg', f"{student_data['current_avg']:.1f}")
        with col4:
            risk_color = '🔴' if student_data['final_risk'] == 'High' else '🟡' if student_data['final_risk'] == 'Medium' else '🟢'
            st.metric('Risk Level', f"{risk_color} {student_data['final_risk']}")

        # Detailed Information
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('📋 Student Details')

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {student_data['name']}")
            st.write(f"**Student ID:** {student_data['student_id']}")
            st.write(f"**Guardian Email:** {student_data['guardian_email']}")
            st.write(f"**Late Payments:** {int(student_data['late_payments'])}")

        with col2:
            st.write(f"**Attendance:** {student_data['attended_classes']}/{student_data['total_classes']} ({student_data['attendance_percent']:.1f}%)")
            st.write(f"**Current Average:** {student_data['current_avg']:.1f}")
            st.write(f"**Previous Average:** {student_data['previous_avg']:.1f}")
            st.write(f"**Score Trend:** {'📈' if student_data['score_trend'] > 0 else '📉'} {student_data['score_trend']:.1f}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Performance Charts
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('📊 Performance Analysis')

        # Create sample historical data for demonstration
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        attendance_history = [85, 78, student_data['attendance_percent'], 82, 88, 91]
        scores_history = [student_data['previous_avg'] - 5, student_data['previous_avg'] - 2, student_data['previous_avg'], student_data['current_avg'] - 3, student_data['current_avg'], student_data['current_avg'] + 2]

        col1, col2 = st.columns(2)
        with col1:
            fig_attendance = px.line(x=months, y=attendance_history, title='Attendance Trend',
                                   labels={'x': 'Month', 'y': 'Attendance %'})
            fig_attendance.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Minimum Required")
            st.plotly_chart(fig_attendance, use_container_width=True)

        with col2:
            fig_scores = px.line(x=months, y=scores_history, title='Academic Performance Trend',
                               labels={'x': 'Month', 'y': 'Average Score'})
            fig_scores.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Passing Grade")
            st.plotly_chart(fig_scores, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Risk Assessment
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('🎯 Risk Assessment')

        # Risk factors analysis
        risk_factors = []
        if student_data['attendance_percent'] < 75:
            risk_factors.append("Low attendance")
        if student_data['current_avg'] < 60:
            risk_factors.append("Poor academic performance")
        if student_data['late_payments'] > 0:
            risk_factors.append("Outstanding fees")
        if student_data['score_trend'] < -5:
            risk_factors.append("Declining performance trend")

        if risk_factors:
            st.error("⚠️ **Risk Factors Identified:**")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.success("✅ **No major risk factors detected**")

        # Recommendations
        st.subheader('💡 AI Recommendations')
        if student_data['final_risk'] == 'High':
            st.warning("**Immediate Actions Required:**")
            st.write("• Schedule counseling session within 3 days")
            st.write("• Contact guardian for parent-teacher meeting")
            st.write("• Provide additional academic support")
            st.write("• Monitor attendance closely")
        elif student_data['final_risk'] == 'Medium':
            st.info("**Recommended Actions:**")
            st.write("• Regular progress monitoring")
            st.write("• Additional study materials if needed")
            st.write("• Parent communication")
        else:
            st.success("**Positive Indicators:**")
            st.write("• Continue current academic performance")
            st.write("• Maintain attendance levels")
            st.write("• Consider advanced learning opportunities")

        st.markdown('</div>', unsafe_allow_html=True)

def show_interventions():
    """Interventions Management Page"""
    st.header('🎯 Intervention Management')

    if 'processed_df' not in st.session_state or st.session_state.processed_df.empty:
        st.warning('Please run prediction on the Dashboard first.')
        return

    processed_df = st.session_state.processed_df

    # Intervention tracking
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('📝 Active Interventions')

    # Sample intervention data (in real app, this would come from database)
    interventions = [
        {'student': 'John Doe', 'type': 'Counseling Session', 'status': 'Scheduled', 'date': '2024-01-15'},
        {'student': 'Jane Smith', 'type': 'Academic Support', 'status': 'In Progress', 'date': '2024-01-10'},
        {'student': 'Bob Johnson', 'type': 'Parent Meeting', 'status': 'Completed', 'date': '2024-01-08'}
    ]

    if interventions:
        intervention_df = pd.DataFrame(interventions)
        st.dataframe(intervention_df)
    else:
        st.info('No active interventions at the moment.')

    st.markdown('</div>', unsafe_allow_html=True)

    # Add new intervention
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('➕ Add New Intervention')

    high_risk_students = processed_df[processed_df['final_risk'] == 'High']['name'].tolist()

    with st.form('intervention_form'):
        student_name = st.selectbox('Select Student', high_risk_students)
        intervention_type = st.selectbox('Intervention Type',
                                       ['Counseling Session', 'Academic Support', 'Parent Meeting',
                                        'Financial Aid', 'Attendance Monitoring', 'Study Group'])
        priority = st.selectbox('Priority', ['High', 'Medium', 'Low'])
        notes = st.text_area('Notes/Comments')
        submit = st.form_submit_button('Schedule Intervention')

        if submit:
            st.success(f'✅ Intervention scheduled for {student_name}: {intervention_type}')
            # In real app, save to database

    st.markdown('</div>', unsafe_allow_html=True)

def show_analytics():
    """Analytics and Reports Page"""
    st.header('📈 Advanced Analytics')

    if 'processed_df' not in st.session_state or st.session_state.processed_df.empty:
        st.warning('Please run prediction on the Dashboard first.')
        return

    processed_df = st.session_state.processed_df

    # Trend Analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('📊 Trend Analysis')

    col1, col2 = st.columns(2)
    with col1:
        # Risk distribution pie chart
        risk_counts = processed_df['final_risk'].value_counts()
        fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index,
                        title='Risk Distribution', color=risk_counts.index,
                        color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Performance correlation
        fig_scatter = px.scatter(processed_df, x='attendance_percent', y='current_avg',
                               color='final_risk', title='Attendance vs Academic Performance',
                               color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Export functionality
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('📤 Export Reports')

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('📊 Export Risk Analysis (CSV)'):
            csv_data = processed_df.to_csv(index=False)
            st.download_button('Download CSV', csv_data, 'risk_analysis.csv', 'text/csv')

    with col2:
        if st.button('📈 Export Summary Report (PDF)'):
            st.info('PDF export functionality would be implemented here')

    with col3:
        if st.button('📋 Export Student List (Excel)'):
            st.info('Excel export functionality would be implemented here')

    st.markdown('</div>', unsafe_allow_html=True)

def show_settings():
    """Settings Page"""
    st.header('⚙️ Settings')

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('📧 Email Configuration')

    st.info('Configure your email settings in the .env file:')
    st.code('EMAIL_USER=your.email@example.com\nEMAIL_PASS=your_app_password')

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('📱 WhatsApp Configuration')

    st.info('Configure your Twilio WhatsApp settings in the .env file:')
    st.code('TWILIO_ACCOUNT_SID=your_account_sid\nTWILIO_AUTH_TOKEN=your_auth_token\nTWILIO_WHATSAPP_NUMBER=whatsapp:+1234567890')

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('🔧 System Settings')

    # Model settings
    st.subheader('🤖 Model Configuration')
    model_depth = st.slider('Decision Tree Max Depth', min_value=3, max_value=10, value=4)
    st.write(f'Current max depth: {model_depth}')

    # Alert settings
    st.subheader('🚨 Alert Settings')
    auto_email = st.checkbox('Enable automatic email alerts for high-risk students', value=True)
    auto_whatsapp = st.checkbox('Enable automatic WhatsApp alerts for high-risk students', value=False)
    alert_threshold = st.selectbox('Alert threshold', ['High Risk Only', 'Medium & High Risk', 'All Risk Levels'])

    st.markdown('</div>', unsafe_allow_html=True)



# --------- Helpers ---------

def load_csv(uploaded_file, sample_path=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, comment='#')
            if not df.empty and 'student_id' in df.columns:
                df['student_id'] = df['student_id'].astype(int)
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return pd.DataFrame()
    else:
        if sample_path and os.path.exists(sample_path):
            try:
                df = pd.read_csv(sample_path, comment='#')
                if not df.empty and 'student_id' in df.columns:
                    df['student_id'] = df['student_id'].astype(int)
                return df
            except pd.errors.EmptyDataError:
                st.error(f"Sample file {sample_path} is empty or invalid.")
                return pd.DataFrame()
            except Exception as e:
                st.error(f"Error loading sample file {sample_path}: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

def preprocess(df):
    # Check required columns
    required_columns = ['student_id', 'name', 'attended_classes', 'total_classes', 'score', 'previous_score', 'late_payments', 'guardian_email']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}")
        return pd.DataFrame()

    # Attendance percent
    df['attendance_percent'] = np.where(df['total_classes'] != 0,
                                       (df['attended_classes'] / df['total_classes']) * 100,
                                       0)

    # Marks aggregation
    marks_agg = df.groupby('student_id').agg(
        current_avg=('score', 'mean'),
        previous_avg=('previous_score', 'mean')
    ).reset_index()

    # Fees aggregation
    fees_agg = df.groupby('student_id').agg(
        late_payments=('late_payments', 'max'),
        guardian_email=('guardian_email', 'first'),
        guardian_phone=('guardian_phone', 'first')
    ).reset_index()

    # Merge all
    merged_df = df[['student_id', 'name', 'attended_classes', 'total_classes', 'attendance_percent']].drop_duplicates(subset=['student_id'])
    merged_df = merged_df.merge(marks_agg, on='student_id', how='left')
    merged_df = merged_df.merge(fees_agg, on='student_id', how='left')

    # Fill missing
    merged_df['current_avg'] = merged_df['current_avg'].fillna(0)
    merged_df['previous_avg'] = merged_df['previous_avg'].fillna(merged_df['current_avg'])
    merged_df['late_payments'] = merged_df['late_payments'].fillna(0)
    merged_df['attendance_percent'] = merged_df['attendance_percent'].fillna(0)

    # Score trend
    merged_df['score_trend'] = merged_df['current_avg'] - merged_df['previous_avg']
    return merged_df

def create_rule_labels(df):
    def assign_label(row):
        if (row['attendance_percent'] < 50) or (row['current_avg'] < 35) or (row['late_payments'] > 2):
            return 2  # High
        elif (row['attendance_percent'] < 80) or (row['current_avg'] < 60) or (row['late_payments'] > 0):
            return 1  # Medium
        else:
            return 0  # Low
    df['label'] = df.apply(assign_label, axis=1)
    return df

def train_and_save_model(df, model_path=MODEL_PATH):
    features = ['attendance_percent', 'current_avg', 'score_trend', 'late_payments']
    X = df[features].fillna(0)
    y = df['label']
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    return clf

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def predict_with_model(model, df):
    features = ['attendance_percent', 'current_avg', 'score_trend', 'late_payments']
    X = df[features].fillna(0)
    df['model_pred'] = model.predict(X)

    def pred2cat(pred):
        if pred == 2:
            return 'High'
        elif pred == 1:
            return 'Medium'
        else:
            return 'Low'

    df['predicted_risk'] = df['model_pred'].apply(pred2cat)
    return df

def apply_rule_engine(df):
    if 'predicted_risk' not in df.columns:
        df['predicted_risk'] = 'Low'  # default if missing

    def rule(row):
        if (row['attendance_percent'] < 50) or (row['current_avg'] < 35) or (row['late_payments'] > 2):
            return 'High'
        elif (row['attendance_percent'] < 80) or (row['current_avg'] < 60) or (row['late_payments'] > 0):
            return 'Medium'
        else:
            return 'Low'

    df['rule_risk'] = df.apply(rule, axis=1)
    rank = {'Low': 0, 'Medium': 1, 'High': 2}
    df['final_risk'] = df.apply(
        lambda r: r['rule_risk'] if rank[r['rule_risk']] >= rank[r['predicted_risk']] else r['predicted_risk'],
        axis=1
    )
    return df

def merge_csvs(attendance_df, marks_df, fees_df):
    if attendance_df.empty or marks_df.empty or fees_df.empty:
        st.error("One or more CSV files are empty.")
        return pd.DataFrame()

    # Merge attendance and marks on student_id
    merged_df = attendance_df.merge(marks_df, on='student_id', how='outer')
    # Drop 'name' from fees_df to avoid column conflict
    if 'name' in fees_df.columns:
        fees_df = fees_df.drop('name', axis=1)
    # Merge with fees
    merged_df = merged_df.merge(fees_df, on='student_id', how='outer')

    # Fill missing values
    merged_df['name'] = merged_df['name'].fillna('Unknown')
    merged_df['attended_classes'] = merged_df['attended_classes'].fillna(0)
    merged_df['total_classes'] = merged_df['total_classes'].fillna(0)
    merged_df['score'] = merged_df['score'].fillna(0)
    merged_df['previous_score'] = merged_df['previous_score'].fillna(0)
    merged_df['late_payments'] = merged_df['late_payments'].fillna(0)
    merged_df['guardian_email'] = merged_df['guardian_email'].fillna('')
    if 'guardian_phone' in merged_df.columns:
        merged_df['guardian_phone'] = merged_df['guardian_phone'].fillna('')
    else:
        merged_df['guardian_phone'] = ''

    return merged_df

def send_email(to_email, subject, body, smtp_server=None, smtp_port=None):
    EMAIL_USER = os.getenv('EMAIL_USER')
    EMAIL_PASS = os.getenv('EMAIL_PASS')

    if not EMAIL_USER or not EMAIL_PASS:
        st.error("❌ EMAIL_USER or EMAIL_PASS not found in .env file.")
        return False

    # Auto-detect SMTP if not provided
    if not smtp_server:
        if EMAIL_USER.endswith("@gmail.com"):
            smtp_server, smtp_port = "smtp.gmail.com", 465
        elif EMAIL_USER.endswith("@outlook.com") or EMAIL_USER.endswith("@paruluniversity.ac.in"):
            smtp_server, smtp_port = "smtp.office365.com", 587
        else:
            smtp_server, smtp_port = "smtp.gmail.com", 465  # default fallback

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        if smtp_port == 465:  # Gmail SSL
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:  # Outlook / Office365 TLS
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()

        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True

    except smtplib.SMTPAuthenticationError as auth_err:
        st.error("❌ Authentication failed! Use App Password (for Gmail enable 2FA, Outlook/Parul mail use given password).")
        st.error(f"Authentication error details: {auth_err}")
        return False
    except smtplib.SMTPConnectError as conn_err:
        st.error("❌ Connection failed! Check your internet/firewall.")
        st.error(f"Connection error details: {conn_err}")
        return False
    except Exception as e:
        st.error(f"Unexpected email error: {e}")
        return False

def send_whatsapp_message(to_number, message):
    """Send WhatsApp message using Twilio"""
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER')

    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER]):
        st.error("❌ Twilio credentials not found in .env file. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_NUMBER.")
        return False

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=f'whatsapp:{to_number}'
        )
        return True
    except Exception as e:
        st.error(f"❌ Failed to send WhatsApp message: {e}")
        return False


# --------- UI ---------

# Navigation
st.sidebar.title('🎓 Navigation')
page = st.sidebar.radio('Go to', ['📊 Dashboard', '👤 Student Profiles', '🎯 Interventions', '📈 Analytics', '⚙️ Settings'])

st.sidebar.header('1) Upload data (separate CSVs or combined)')
upload_option = st.sidebar.radio('Upload option', ['Separate CSVs', 'Combined CSV'])

if upload_option == 'Separate CSVs':
    attendance_file = st.sidebar.file_uploader('Upload Attendance CSV', type=['csv'], key='attendance')
    marks_file = st.sidebar.file_uploader('Upload Marks CSV', type=['csv'], key='marks')
    fees_file = st.sidebar.file_uploader('Upload Fees CSV', type=['csv'], key='fees')
    use_sample = st.sidebar.checkbox('Use sample data (from data/ folder)', value=True)

    if use_sample:
        attendance_df = load_csv(attendance_file, os.path.join(SAMPLE_DIR, 'sample_attendance.csv'))
        marks_df = load_csv(marks_file, os.path.join(SAMPLE_DIR, 'sample_marks.csv'))
        fees_df = load_csv(fees_file, os.path.join(SAMPLE_DIR, 'sample_fees.csv'))
    else:
        attendance_df = load_csv(attendance_file)
        marks_df = load_csv(marks_file)
        fees_df = load_csv(fees_file)

    # Merge separate CSVs
    df = merge_csvs(attendance_df, marks_df, fees_df)
else:
    data_file = st.sidebar.file_uploader('Upload combined CSV', type=['csv'])
    use_sample = st.sidebar.checkbox('Use sample data (from data/ folder)', value=True)

    if use_sample:
        df = load_csv(data_file, os.path.join(SAMPLE_DIR, 'sample_combined.csv'))
    else:
        df = load_csv(data_file)

if df.empty:
    st.warning('Data missing — sample will be used if available.')

if st.sidebar.button('Prepare & Train Model (quick)'):
    processed_df = preprocess(df)
    if not processed_df.empty:
        processed_df = create_rule_labels(processed_df)
        clf = train_and_save_model(processed_df)
        if clf:
            st.success('Model trained and saved to models/risk_model.pkl')
        else:
            st.error("Failed to train model.")
    else:
        st.error("Preprocessing failed.")

model = load_model()

if st.sidebar.button('Run Prediction'):
    if model is None:
        st.error('Model not found. Pehle "Prepare & Train Model" karlo.')
    else:
        processed_df = preprocess(df)
        if not processed_df.empty:
            processed_df = predict_with_model(model, processed_df)
            processed_df = apply_rule_engine(processed_df)

            # Metrics Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card"><div class="metric-value">' + str(len(processed_df)) + '</div><div class="metric-label">Total Students</div></div>', unsafe_allow_html=True)
            with col2:
                high_count = len(processed_df[processed_df['final_risk'] == 'High'])
                st.markdown('<div class="metric-card"><div class="metric-value">' + str(high_count) + '</div><div class="metric-label">High Risk</div></div>', unsafe_allow_html=True)
            with col3:
                medium_count = len(processed_df[processed_df['final_risk'] == 'Medium'])
                st.markdown('<div class="metric-card"><div class="metric-value">' + str(medium_count) + '</div><div class="metric-label">Medium Risk</div></div>', unsafe_allow_html=True)
            with col4:
                low_count = len(processed_df[processed_df['final_risk'] == 'Low'])
                st.markdown('<div class="metric-card"><div class="metric-value">' + str(low_count) + '</div><div class="metric-label">Low Risk</div></div>', unsafe_allow_html=True)

            # Prediction Results
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader('📊 Prediction Results')
            display_df = processed_df[['student_id', 'name', 'attendance_percent', 'current_avg',
                                       'late_payments', 'predicted_risk', 'rule_risk',
                                       'final_risk', 'guardian_email']].sort_values(by='final_risk', ascending=False)
            page_size = st.slider('Rows per page', min_value=10, max_value=100, value=20, step=10)
            st.dataframe(display_df.head(page_size))
            st.markdown('</div>', unsafe_allow_html=True)

            # Risk Distribution Chart
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader('📈 Risk Distribution')
            counts = processed_df['final_risk'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
            fig = px.bar(x=counts.index, y=counts.values, labels={'x': 'Risk Level', 'y': 'Count'},
                        title='Students per Risk Level', color=counts.index,
                        color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Risk Indicator
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader('🎯 Risk Indicator')
            def risk_color(risk):
                if risk == 'High':
                    return '🔴 High Risk'
                elif risk == 'Medium':
                    return '🟡 Medium Risk'
                else:
                    return '🟢 Low Risk'

            processed_df['risk_indicator'] = processed_df['final_risk'].apply(risk_color)
            risk_display_df = processed_df[['name', 'risk_indicator']].sort_values(by='risk_indicator', ascending=False)
            st.dataframe(risk_display_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Email Alerts
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader('📧 Send Email Alerts')
            st.info('📝 To send emails, set EMAIL_USER and EMAIL_PASS in .env file. For Gmail, use app password (not regular password). Enable 2FA and generate app password at https://myaccount.google.com/apppasswords')
            smtp_server = st.text_input('SMTP Server', value='smtp.gmail.com', placeholder='smtp.gmail.com')
            smtp_port = st.number_input('SMTP Port', value=465, min_value=1, max_value=65535)
            test_email = st.text_input('Test Email Address', placeholder='your.email@example.com')
            if st.button('📤 Send Test Email'):
                if test_email:
                    subject = "Test Email from Dropout Prediction System"
                    body = "This is a test email to verify email configuration."
                    if send_email(test_email, subject, body, smtp_server, smtp_port):
                        st.success('✅ Test email sent successfully!')
                    else:
                        st.error('❌ Failed to send test email. Check .env, SMTP settings, and credentials.')
                else:
                    st.warning('Please enter a test email address.')
            high_df = processed_df[processed_df['final_risk'] == 'High']
            st.write(f'🔴 High risk students: {len(high_df)}')
            missing_emails = high_df['guardian_email'].isna() | (high_df['guardian_email'] == '')
            missing_count = missing_emails.sum()
            if missing_count > 0:
                st.warning(f'⚠️ {missing_count} students have missing guardian emails and will be skipped.')
            success = 0
            if len(high_df) > 0:
                if st.button('🚀 Send Emails to Guardians', key='send_emails'):
                    for _, row in high_df.iterrows():
                        to = row.get('guardian_email', '')
                        if pd.isna(to) or to == '':
                            continue
                        subject = f"Academic Performance Alert: {row['name']} - Student at Risk of Dropout"
                        body = (
                            f"Dear Guardian,\n\n"
                            f"Your ward {row['name']} (ID: {row['student_id']}) has been identified as HIGH RISK for dropout.\n\n"
                            f"Details:\n"
                            f"- Attendance: {row['attendance_percent']:.1f}%\n"
                            f"- Average Score: {row['current_avg']:.1f}\n"
                            f"- Late Payments: {int(row['late_payments'])}\n\n"
                            f"Please contact the school counselor immediately for support and guidance.\n\n"
                            f"Best regards,\n"
                            f"School Administration\n"
                            f"AI Dropout Prediction System"
                        )
                        try:
                            if send_email(to, subject, body, smtp_server, smtp_port):
                                success += 1
                        except Exception as e:
                            st.error(f"Failed to send email to {to}: {e}")
                    st.success(f'✅ Emails sent: {success}/{len(high_df) - missing_count} (skipped {missing_count} with missing emails)')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Preprocessing failed.")

# Integration: Send emails automatically after prediction for high risk students with valid guardian emails
if 'processed_df' in locals() and not processed_df.empty:
    if 'final_risk' not in processed_df.columns:
        processed_df = apply_rule_engine(processed_df)
    high_risk_students = processed_df[(processed_df['final_risk'] == 'High') & (processed_df['guardian_email'] != '') & (~processed_df['guardian_email'].isna())]
    if not high_risk_students.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('🚨 Automatic Email Alerts for High Risk Students')
        sent_count = 0
        for _, row in high_risk_students.iterrows():
            to = row['guardian_email']
            subject = f"Academic Performance Alert: {row['name']} - Student at Risk of Dropout"
            body = (
                f"Dear Guardian,\n\n"
                f"Your ward {row['name']} (ID: {row['student_id']}) has been identified as HIGH RISK for dropout.\n\n"
                f"Details:\n"
                f"- Attendance: {row['attendance_percent']:.1f}%\n"
                f"- Average Score: {row['current_avg']:.1f}\n"
                f"- Late Payments: {int(row['late_payments'])}\n\n"
                f"Please contact the school counselor immediately for support and guidance.\n\n"
                f"Best regards,\n"
                f"School Administration\n"
                f"AI Dropout Prediction System"
            )
            if send_email(to, subject, body):
                sent_count += 1
        st.success(f'✅ Automatically sent emails to {sent_count} high risk students\' guardians.')
        st.markdown('</div>', unsafe_allow_html=True)

st.info('Tips: Upload a single combined CSV with all required columns or use sample data. Train model first, then run prediction.')

# Automatic WhatsApp sending after prediction for high risk students with valid guardian phone numbers
if 'processed_df' in locals() and not processed_df.empty:
    if 'final_risk' not in processed_df.columns:
        processed_df = apply_rule_engine(processed_df)
    high_risk_students_whatsapp = processed_df[(processed_df['final_risk'] == 'High') & (processed_df['guardian_phone'] != '') & (~processed_df['guardian_phone'].isna())]
    if not high_risk_students_whatsapp.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader('🚨 Automatic WhatsApp Alerts for High Risk Students')
        sent_count_whatsapp = 0
        for _, row in high_risk_students_whatsapp.iterrows():
            to = row['guardian_phone']
            subject = f"Academic Performance Alert: {row['name']} - Student at Risk of Dropout"
            message = (
                f"Dear Guardian,\n\n"
                f"Your ward {row['name']} (ID: {row['student_id']}) has been identified as HIGH RISK for dropout.\n\n"
                f"Details:\n"
                f"- Attendance: {row['attendance_percent']:.1f}%\n"
                f"- Average Score: {row['current_avg']:.1f}\n"
                f"- Late Payments: {int(row['late_payments'])}\n\n"
                f"Please contact the school counselor immediately for support and guidance.\n\n"
                f"Best regards,\n"
                f"School Administration\n"
                f"AI Dropout Prediction System"
            )
            if send_whatsapp_message(to, message):
                sent_count_whatsapp += 1
        st.success(f'✅ Automatically sent WhatsApp messages to {sent_count_whatsapp} high risk students\' guardians.')
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Smart India Hackathon — SIH25102 | AI-Powered Education Solutions</p>
    <p>Built with ❤️ using Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# --------- Main Page Logic ---------

if page == '📊 Dashboard':
    show_dashboard()
elif page == '👤 Student Profiles':
    show_student_profiles()
elif page == '🎯 Interventions':
    show_interventions()
elif page == '📈 Analytics':
    show_analytics()
elif page == '⚙️ Settings':
    show_settings()
