import pyrebase
import streamlit as st

# Firebase configuration
firebase_config = {
    "apiKey": "AIzaSyCBaVJwy-zhpfMkkl-2BHyHS6tRzwVODlg",
    "authDomain": "salesforecastingstream.firebaseapp.com",
    "databaseURL": "https://salesforecastingstream.firebaseio.com",
    "projectId": "salesforecastingstream",
    "storageBucket": "salesforecastingstream.firebasestorage.app",
    "messagingSenderId": "49098940237",
    "appId": "1:49098940237:web:2027584f9d77771126a104"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

def sign_up(email, password, name):
    if not email or not password or not name:
        st.error("All fields are required.")
        return
    if len(password) < 6:
        st.error("🔒 Password must be at least 6 characters long.")
        return
    try:
        user = auth.create_user_with_email_and_password(email, password)
        st.session_state['user'] = {'email': email, 'name': name}
        st.session_state['authenticated'] = True
        st.success(f"🎉 Welcome {name}, you have successfully signed up!")
    except Exception as e:
        error_message = str(e)
        if "EMAIL_EXISTS" in error_message:
            st.error("📧 This email is already registered. Try logging in or resetting your password.")
        else:
            st.error(f"⚠️ An unexpected error occurred: {error_message}")


# Login
def login(email, password):
    if not email or not password:
        st.error("Please enter both email and password.")
        return
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.session_state['user'] = {'email': email}
        st.session_state['authenticated'] = True
        st.success("🎉 Welcome back!")
    except Exception as e:
        st.session_state['authenticated'] = False
        st.session_state['user'] = None
        error_message = str(e)
        if "INVALID_LOGIN_CREDENTIALS" in error_message or "INVALID_PASSWORD" in error_message:
            st.error("❌ Invalid email or password.")
            st.info("🔐 Need help? Use the 'Forgot Password?' option.")
        else:
            st.error(f"Unexpected error: {error_message}")


# Password reset
def reset_password(email):
    if not email:
        st.error("Please enter your email.")
        return
    try:
        auth.send_password_reset_email(email)
        st.success("✅ Password reset email sent! Check your inbox.")
    except Exception as e:
        st.error(f"Error sending reset email: {e}")


# Main auth UI
def show_login_signup():
    st.title("🔐 Login / Signup")
    tabs = st.tabs(["Login", "Signup"])

    # --- LOGIN TAB ---
    with tabs[0]:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            login(email, password)

        if st.button("Forgot Password?"):
            st.session_state["show_reset"] = True

        if st.session_state.get("show_reset"):
            reset_email = st.text_input("Enter your email to reset password", key="reset_email")
            if st.button("Send Reset Link"):
                reset_password(reset_email)

    # --- SIGNUP TAB ---
    with tabs[1]:
        name = st.text_input("Name", key="signup_name")
        email_signup = st.text_input("Email", key="signup_email")
        password_signup = st.text_input("Password", type="password", key="signup_password")
        if st.button("Signup"):
            sign_up(email_signup, password_signup, name)

def logout():
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
    st.success("You have logged out successfully.")
