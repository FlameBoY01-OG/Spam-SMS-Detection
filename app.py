import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# Load models
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        return None, None


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .spam-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border: 2px solid #ff4757;
    }

    .not-spam-box {
        background: linear-gradient(135deg, #2ed573, #1e90ff);
        color: white;
        border: 2px solid #2ed573;
    }

    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }

    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }

    .stat-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
    }

    .example-messages {
        background: #2b2b2b;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Smart Spam Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Protect yourself from spam messages with AI-powered detection</p>',
                unsafe_allow_html=True)

    # Load models
    tfidf, model = load_models()

    if tfidf is None or model is None:
        st.stop()

    # Main content area
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📝 Enter Your Message")

        # Text input with placeholder
        input_sms = st.text_area(
            "",
            placeholder="Type or paste your email/SMS message here...",
            height=150,
            help="Enter the complete message you want to classify"
        )

        # Character count
        if input_sms:
            char_count = len(input_sms)
            word_count = len(input_sms.split())
            st.caption(f"📏 Characters: {char_count} | Words: {word_count}")

        # Prediction button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button(
                "🔍 Analyze Message",
                type="primary",
                use_container_width=True
            )

        # Clear button
        with col_btn3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()

    with col2:
        st.subheader("📋 Example Messages")

        # Example messages
        st.markdown("""
        <div class="example-messages">
        <strong>💰 Typical Spam:</strong><br>
        "Congratulations! You've won $1000! Click here to claim your prize now!"<br><br>
        <strong>✅ Legitimate Message:</strong><br>
        "Hi! Hope you're doing well. Let's catch up over coffee this weekend."
        </div>
        """, unsafe_allow_html=True)

        # Quick test buttons
        st.subheader("🚀 Quick Test")
        spam_example = "URGENT! You've won $5000! Click this link immediately to claim your prize before it expires!"
        normal_example = "Hey, just wanted to remind you about our meeting tomorrow at 3 PM. See you then!"

        if st.button("Test Spam Example", use_container_width=True):
            st.session_state.test_message = spam_example
            st.rerun()

        if st.button("Test Normal Example", use_container_width=True):
            st.session_state.test_message = normal_example
            st.rerun()

    # Handle test messages
    if hasattr(st.session_state, 'test_message'):
        input_sms = st.session_state.test_message
        predict_button = True
        del st.session_state.test_message

    # Prediction logic
    if predict_button and input_sms:
        with st.spinner("🤖 Analyzing message..."):
            # Add a small delay for better UX
            time.sleep(0.5)

            try:
                # 1. Preprocess
                transformed_sms = transform_text(input_sms)

                # 2. Vectorize
                vector_input = tfidf.transform([transformed_sms])

                # 3. Predict
                result = model.predict(vector_input)[0]

                # 4. Display Results
                st.markdown("---")
                st.subheader("🎯 Classification Result")

                if result == 1:
                    st.markdown(f"""
                    <div class="prediction-box spam-box">
                        🚨 SPAM DETECTED 🚨<br>
                        This message appears to be spam
                    </div>
                    """, unsafe_allow_html=True)

                    st.warning(
                        "⚠️ **Warning**: This message has characteristics commonly found in spam. Be cautious about:")
                    st.markdown(
                        "• Clicking any links\n• Sharing personal information\n• Responding to unknown senders\n• Prize/money claims")

                else:
                    st.markdown(f"""
                    <div class="prediction-box not-spam-box">
                        ✅ LEGITIMATE MESSAGE ✅<br>
                        This message appears to be safe
                    </div>
                    """, unsafe_allow_html=True)

                    st.success("✅ **Good News**: This message appears to be legitimate and safe.")

                # Show processed text in expander
                with st.expander("🔍 View Processed Text"):
                    st.code(transformed_sms, language="text")
                    st.caption(
                        "This is how the message looks after preprocessing (lowercased, stemmed, stop words removed)")

            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {str(e)}")

    elif predict_button and not input_sms:
        st.warning("📝 Please enter a message to analyze!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🛡️ <strong>Stay Safe Online</strong> | Always verify suspicious messages through official channels</p>
        <p>Made with ❤️ using Streamlit and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()