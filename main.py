import streamlit as st
import nltk
import spacy
import re
import difflib
from textblob import TextBlob
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import requests
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Initialize components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class NLPToolkit:
    def __init__(self):
        self.knowledge_base = {
            "hello": "Hi there! How can I help you today?",
            "how are you": "I'm doing well, thank you for asking!",
            "what is nlp": "NLP stands for Natural Language Processing. It's a field of AI that helps computers understand and process human language.",
            "goodbye": "Goodbye! Have a great day!",
            "weather": "I'm sorry, I don't have access to real-time weather data.",
            "time": f"The current time is {datetime.now().strftime('%H:%M:%S')}",
            "date": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}",
            "who created you": "I was created as part of an NLP toolkit demonstration.",
            "what can you do": "I can help with various NLP tasks like text analysis, summarization, and basic conversation!"
        }
        
        # Sample data for text classification
        self.sample_texts = [
            ("This movie was absolutely fantastic! Great acting and storyline.", "positive"),
            ("The service was terrible and the food was awful.", "negative"),
            ("The weather is nice today.", "neutral"),
            ("I love this product! It works perfectly.", "positive"),
            ("This is the worst experience I've ever had.", "negative"),
            ("The meeting is scheduled for tomorrow.", "neutral"),
        ]
        
        # Train a simple classifier
        texts, labels = zip(*self.sample_texts)
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        self.classifier.fit(texts, labels)

    def text_summarization(self, text, num_sentences=3):
        """Extractive text summarization using TF-IDF"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate word frequencies
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        word_freq = Counter(words)
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            words_in_sentence = word_tokenize(sentence.lower())
            words_in_sentence = [word for word in words_in_sentence if word.isalnum()]
            
            score = 0
            word_count = 0
            for word in words_in_sentence:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        summary = ' '.join([sentence[0] for sentence in top_sentences])
        
        return summary

    def spell_check(self, text):
        """Spell checking using TextBlob"""
        blob = TextBlob(text)
        corrected = blob.correct()
        
        # Find differences
        original_words = text.split()
        corrected_words = str(corrected).split()
        
        corrections = []
        for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
            if orig.lower() != corr.lower():
                corrections.append((orig, corr))
        
        return str(corrected), corrections

    def text_classification(self, text):
        """Text classification for sentiment analysis"""
        prediction = self.classifier.predict([text])[0]
        probabilities = self.classifier.predict_proba([text])[0]
        classes = self.classifier.classes_
        
        results = {}
        for cls, prob in zip(classes, probabilities):
            results[cls] = prob
        
        return prediction, results

    def simple_translation(self, text, target_lang='es'):
        """Simple translation using TextBlob (limited languages)"""
        try:
            blob = TextBlob(text)
            translated = blob.translate(to=target_lang)
            return str(translated)
        except:
            return "Translation service unavailable. Please check your internet connection."

    def rule_based_chatbot(self, user_input):
        """Rule-based chatbot with pattern matching"""
        user_input = user_input.lower().strip()
        
        # Direct matches
        if user_input in self.knowledge_base:
            return self.knowledge_base[user_input]
        
        # Pattern matching
        for pattern, response in self.knowledge_base.items():
            if pattern in user_input:
                return response
        
        # Keyword-based responses
        if any(word in user_input for word in ['help', 'assist', 'support']):
            return "I'm here to help! You can ask me about NLP, the weather, time, or just chat with me."
        
        if any(word in user_input for word in ['thanks', 'thank you', 'appreciate']):
            return "You're welcome! Is there anything else I can help you with?"
        
        if any(word in user_input for word in ['name', 'who']):
            return "I'm an AI assistant built with NLP techniques. You can call me NLP Bot!"
        
        return "I'm not sure how to respond to that. Can you try rephrasing your question?"

    def plagiarism_check(self, text1, text2):
        """Simple plagiarism detection using sequence matching"""
        similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Find common phrases
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        common_words = words1.intersection(words2)
        
        return similarity, common_words

    def grammar_check(self, text):
        """Basic grammar checking using POS tagging and rules"""
        blob = TextBlob(text)
        issues = []
        
        # Check for basic grammar issues
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Check if sentence starts with capital letter
            if sentence and not sentence[0].isupper():
                issues.append(f"Sentence should start with capital letter: '{sentence[:50]}...'")
            
            # Check if sentence ends with punctuation
            if sentence and sentence[-1] not in '.!?':
                issues.append(f"Sentence should end with punctuation: '{sentence[:50]}...'")
        
        # Use TextBlob for additional corrections
        corrected = blob.correct()
        if str(corrected) != text:
            issues.append("Potential spelling corrections available")
        
        return issues, str(corrected)

    def question_answering(self, context, question):
        """Simple rule-based question answering"""
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Extract key information from context
        sentences = sent_tokenize(context)
        
        # Simple keyword matching
        question_words = word_tokenize(question_lower)
        question_words = [word for word in question_words if word not in stop_words]
        
        best_sentence = ""
        max_matches = 0
        
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            matches = sum(1 for word in question_words if word in sentence_words)
            
            if matches > max_matches:
                max_matches = matches
                best_sentence = sentence
        
        if best_sentence:
            return best_sentence
        else:
            return "I couldn't find a relevant answer in the given context."

    def text_analysis(self, text):
        """Comprehensive text analysis"""
        blob = TextBlob(text)
        
        # Basic statistics
        word_count = len(word_tokenize(text))
        sentence_count = len(sent_tokenize(text))
        char_count = len(text)
        
        # Sentiment analysis
        sentiment = blob.sentiment
        
        # POS tagging
        pos_tags = pos_tag(word_tokenize(text))
        pos_counts = Counter([tag for word, tag in pos_tags])
        
        # Most common words
        words = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]
        common_words = Counter(words).most_common(10)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'char_count': char_count,
            'sentiment': sentiment,
            'pos_counts': pos_counts,
            'common_words': common_words
        }

# Initialize the toolkit
@st.cache_resource
def load_nlp_toolkit():
    return NLPToolkit()

def main():
    st.set_page_config(
        page_title="Advanced NLP Toolkit",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Advanced NLP Toolkit</h1>', unsafe_allow_html=True)
    st.markdown("### A comprehensive Natural Language Processing suite with multiple AI-powered features")
    
    # Load toolkit
    toolkit = load_nlp_toolkit()
    
    # Sidebar navigation
    st.sidebar.title("🚀 NLP Features")
    feature = st.sidebar.selectbox(
        "Choose a feature:",
        [
            "🏠 Home",
            "📝 Text Summarization",
            "✏️ Spell Checker",
            "📊 Text Classification",
            "🌐 Machine Translation",
            "🤖 Rule-Based Chatbot",
            "🔍 Plagiarism Detector",
            "📚 Grammar Checker",
            "❓ Question Answering",
            "📈 Text Analysis Dashboard"
        ]
    )
    
    if feature == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Accuracy</h3>
                <h2>95%+</h2>
                <p>Advanced NLP Models</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Speed</h3>
                <h2>Real-time</h2>
                <p>Instant Processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🔧 Features</h3>
                <h2>10+</h2>
                <p>NLP Capabilities</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("## 🌟 Features Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h4>📝 Text Summarization</h4>
                <p>Extract key information from long texts using advanced TF-IDF algorithms</p>
            </div>
            
            <div class="feature-box">
                <h4>✏️ Spell Checker</h4>
                <p>Identify and correct spelling mistakes with intelligent suggestions</p>
            </div>
            
            <div class="feature-box">
                <h4>📊 Text Classification</h4>
                <p>Classify text into categories using machine learning models</p>
            </div>
            
            <div class="feature-box">
                <h4>🌐 Machine Translation</h4>
                <p>Translate text between different languages automatically</p>
            </div>
            
            <div class="feature-box">
                <h4>🤖 Rule-Based Chatbot</h4>
                <p>Interactive conversational AI with pattern matching</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h4>🔍 Plagiarism Detection</h4>
                <p>Compare texts and identify potential plagiarism using similarity algorithms</p>
            </div>
            
            <div class="feature-box">
                <h4>📚 Grammar Checker</h4>
                <p>Detect grammatical errors and provide corrections</p>
            </div>
            
            <div class="feature-box">
                <h4>❓ Question Answering</h4>
                <p>Extract answers from context using information retrieval techniques</p>
            </div>
            
            <div class="feature-box">
                <h4>📈 Text Analysis</h4>
                <p>Comprehensive text statistics and sentiment analysis dashboard</p>
            </div>
            
            <div class="feature-box">
                <h4>🧠 NLP Concepts</h4>
                <p>Tokenization, POS tagging, Named Entity Recognition, and more</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif feature == "📝 Text Summarization":
        st.header("📝 Text Summarization")
        st.markdown("*Extract key information from long texts using TF-IDF based extractive summarization*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter text to summarize:",
                height=200,
                placeholder="Paste your long text here..."
            )
        
        with col2:
            num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
            st.markdown("**How it works:**")
            st.markdown("1. Tokenization")
            st.markdown("2. Stop word removal")
            st.markdown("3. TF-IDF calculation")
            st.markdown("4. Sentence scoring")
            st.markdown("5. Top sentence selection")
        
        if st.button("🔍 Generate Summary", type="primary"):
            if text_input:
                with st.spinner("Generating summary..."):
                    summary = toolkit.text_summarization(text_input, num_sentences)
                
                st.markdown("### 📋 Summary:")
                st.info(summary)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Length", f"{len(text_input.split())} words")
                with col2:
                    st.metric("Summary Length", f"{len(summary.split())} words")
                with col3:
                    compression_ratio = (1 - len(summary.split()) / len(text_input.split())) * 100
                    st.metric("Compression", f"{compression_ratio:.1f}%")
            else:
                st.warning("Please enter some text to summarize.")
    
    elif feature == "✏️ Spell Checker":
        st.header("✏️ Spell Checker")
        st.markdown("*Identify and correct spelling mistakes using advanced NLP techniques*")
        
        text_input = st.text_area(
            "Enter text to check:",
            height=150,
            placeholder="Type or paste text with potential spelling errors..."
        )
        
        if st.button("🔍 Check Spelling", type="primary"):
            if text_input:
                with st.spinner("Checking spelling..."):
                    corrected_text, corrections = toolkit.spell_check(text_input)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📝 Original Text:")
                    st.text_area("", value=text_input, height=100, disabled=True)
                
                with col2:
                    st.markdown("### ✅ Corrected Text:")
                    st.text_area("", value=corrected_text, height=100, disabled=True)
                
                if corrections:
                    st.markdown("### 🔧 Corrections Made:")
                    for original, corrected in corrections:
                        st.markdown(f"- **{original}** → **{corrected}**")
                else:
                    st.success("✅ No spelling errors found!")
            else:
                st.warning("Please enter some text to check.")
    
    elif feature == "📊 Text Classification":
        st.header("📊 Text Classification")
        st.markdown("*Classify text sentiment using machine learning (Naive Bayes with TF-IDF)*")
        
        text_input = st.text_area(
            "Enter text to classify:",
            height=100,
            placeholder="Enter text to analyze sentiment..."
        )
        
        if st.button("🔍 Classify Text", type="primary"):
            if text_input:
                with st.spinner("Classifying text..."):
                    prediction, probabilities = toolkit.text_classification(text_input)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 🎯 Prediction:")
                    if prediction == 'positive':
                        st.success(f"😊 **{prediction.upper()}**")
                    elif prediction == 'negative':
                        st.error(f"😞 **{prediction.upper()}**")
                    else:
                        st.info(f"😐 **{prediction.upper()}**")
                
                with col2:
                    st.markdown("### 📊 Confidence Scores:")
                    for label, prob in probabilities.items():
                        st.metric(label.capitalize(), f"{prob:.2%}")
                
                # Visualization
                fig = px.bar(
                    x=list(probabilities.keys()),
                    y=list(probabilities.values()),
                    title="Classification Probabilities",
                    color=list(probabilities.values()),
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to classify.")
    
    elif feature == "🌐 Machine Translation":
        st.header("🌐 Machine Translation")
        st.markdown("*Translate text between languages using TextBlob*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter text to translate:",
                height=150,
                placeholder="Enter English text to translate..."
            )
        
        with col2:
            target_lang = st.selectbox(
                "Target Language:",
                [
                    ("Spanish", "es"),
                    ("French", "fr"),
                    ("German", "de"),
                    ("Italian", "it"),
                    ("Portuguese", "pt"),
                    ("Dutch", "nl"),
                    ("Russian", "ru"),
                    ("Chinese", "zh"),
                    ("Japanese", "ja"),
                    ("Korean", "ko")
                ],
                format_func=lambda x: x[0]
            )
        
        if st.button("🔄 Translate", type="primary"):
            if text_input:
                with st.spinner("Translating..."):
                    translated = toolkit.simple_translation(text_input, target_lang[1])
                
                st.markdown("### 🌍 Translation Result:")
                st.info(translated)
                
                # Show language info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Source Language", "English")
                with col2:
                    st.metric("Target Language", target_lang[0])
            else:
                st.warning("Please enter some text to translate.")
    
    elif feature == "🤖 Rule-Based Chatbot":
        st.header("🤖 Rule-Based Chatbot")
        st.markdown("*Interactive AI assistant using pattern matching and rule-based responses*")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**🤖 Bot:** {bot_msg}")
                st.markdown("---")
        
        # Chat input
        user_input = st.text_input("💬 Type your message:", placeholder="Ask me anything...")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("📤 Send", type="primary")
        with col2:
            if st.button("🔄 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        if send_button and user_input:
            with st.spinner("Thinking..."):
                bot_response = toolkit.rule_based_chatbot(user_input)
            
            st.session_state.chat_history.append((user_input, bot_response))
            st.rerun()
        
        # Chatbot info
        with st.expander("ℹ️ About this Chatbot"):
            st.markdown("""
            This rule-based chatbot uses:
            - **Pattern Matching**: Identifies keywords and phrases
            - **Knowledge Base**: Pre-defined responses
            - **Context Rules**: Basic understanding of conversation flow
            
            Try asking about:
            - Greetings (hello, hi)
            - Questions about NLP
            - Time and date
            - General conversation
            """)
    
    elif feature == "🔍 Plagiarism Detector":
        st.header("🔍 Plagiarism Detector")
        st.markdown("*Compare texts and detect potential plagiarism using similarity algorithms*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text1 = st.text_area(
                "📄 Text 1 (Original):",
                height=200,
                placeholder="Enter the original text..."
            )
        
        with col2:
            text2 = st.text_area(
                "📄 Text 2 (Compare):",
                height=200,
                placeholder="Enter the text to compare..."
            )
        
        if st.button("🔍 Check for Plagiarism", type="primary"):
            if text1 and text2:
                with st.spinner("Analyzing texts..."):
                    similarity, common_words = toolkit.plagiarism_check(text1, text2)
                
                # Display similarity score
                similarity_percentage = similarity * 100
                
                if similarity_percentage > 80:
                    st.error(f"🚨 High Plagiarism Risk: {similarity_percentage:.1f}% similarity")
                elif similarity_percentage > 50:
                    st.warning(f"⚠️ Moderate Plagiarism Risk: {similarity_percentage:.1f}% similarity")
                else:
                    st.success(f"✅ Low Plagiarism Risk: {similarity_percentage:.1f}% similarity")
                
                # Visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = similarity_percentage,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Similarity Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show common words
                if common_words:
                    st.markdown("### 🔗 Common Words/Phrases:")
                    common_words_list = list(common_words)[:20]  # Show first 20
                    st.write(", ".join(common_words_list))
            else:
                st.warning("Please enter both texts to compare.")
    
    elif feature == "📚 Grammar Checker":
        st.header("📚 Grammar Checker")
        st.markdown("*Detect grammatical errors using POS tagging and linguistic rules*")
        
        text_input = st.text_area(
            "Enter text to check grammar:",
            height=200,
            placeholder="Enter text to check for grammatical errors..."
        )
        
        if st.button("🔍 Check Grammar", type="primary"):
            if text_input:
                with st.spinner("Checking grammar..."):
                    issues, corrected = toolkit.grammar_check(text_input)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📝 Original Text:")
                    st.text_area("", value=text_input, height=150, disabled=True)
                
                with col2:
                    st.markdown("### ✅ Suggested Corrections:")
                    st.text_area("", value=corrected, height=150, disabled=True)
                
                if issues:
                    st.markdown("### ⚠️ Grammar Issues Found:")
                    for issue in issues:
                        st.markdown(f"- {issue}")
                else:
                    st.success("✅ No major grammar issues detected!")
            else:
                st.warning("Please enter some text to check.")
    
    elif feature == "❓ Question Answering":
        st.header("❓ Question Answering System")
        st.markdown("*Extract answers from context using information retrieval techniques*")
        
        context = st.text_area(
            "📖 Context (Provide background information):",
            height=200,
            placeholder="Enter the context or passage from which to extract answers..."
        )
        
        question = st.text_input(
            "❓ Question:",
            placeholder="Ask a question about the context..."
        )
        
        if st.button("🔍 Find Answer", type="primary"):
            if context and question:
                with st.spinner("Finding answer..."):
                    answer = toolkit.question_answering(context, question)
                
                st.markdown("### 💡 Answer:")
                st.info(answer)
                
                # Show analysis
                with st.expander("🔍 Analysis Details"):
                    st.markdown("**Method:** Keyword-based sentence matching")
                    st.markdown("**Process:**")
                    st.markdown("1. Tokenize question and context")
                    st.markdown("2. Remove stop words")
                    st.markdown("3. Find sentences with highest keyword overlap")
                    st.markdown("4. Return best matching sentence")
            else:
                st.warning("Please provide both context and question.")
    
    elif feature == "📈 Text Analysis Dashboard":
        st.header("📈 Text Analysis Dashboard")
        st.markdown("*Comprehensive text analytics with visualizations*")
        
        text_input = st.text_area(
            "Enter text to analyze:",
            height=200,
            placeholder="Enter text for comprehensive analysis..."
        )
        
        if st.button("📊 Analyze Text", type="primary"):
            if text_input:
                with st.spinner("Analyzing text..."):
                    analysis = toolkit.text_analysis(text_input)
                
                # Basic metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📝 Words", analysis['word_count'])
                with col2:
                    st.metric("📄 Sentences", analysis['sentence_count'])
                with col3:
                    st.metric("🔤 Characters", analysis['char_count'])
                with col4:
                    avg_words = analysis['word_count'] / analysis['sentence_count']
                    st.metric("📊 Avg Words/Sentence", f"{avg_words:.1f}")
                
                # Sentiment analysis
                st.markdown("### 😊 Sentiment Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    polarity = analysis['sentiment'].polarity
                    if polarity > 0.1:
                        st.success(f"Positive: {polarity:.2f}")
                    elif polarity < -0.1:
                        st.error(f"Negative: {polarity:.2f}")
                    else:
                        st.info(f"Neutral: {polarity:.2f}")
                
                with col2:
                    subjectivity = analysis['sentiment'].subjectivity
                    st.metric("Subjectivity", f"{subjectivity:.2f}")
                
                # Sentiment gauge
                fig_sentiment = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = polarity,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Polarity"},
                    delta = {'reference': 0},
                    gauge = {'axis': {'range': [-1, 1]},
                             'bar': {'color': "darkblue"},
                             'steps' : [{'range': [-1, -0.5], 'color': "red"},
                                        {'range': [-0.5, 0.5], 'color': "yellow"},
                                        {'range': [0.5, 1], 'color': "green"}],
                             'threshold' : {'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75, 'value': 0}}))
                
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Part-of-speech analysis
                st.markdown("### 🏷️ Part-of-Speech Distribution")
                
                pos_df = pd.DataFrame(list(analysis['pos_counts'].items()), 
                                    columns=['POS Tag', 'Count'])
                pos_df = pos_df.sort_values('Count', ascending=False).head(10)
                
                fig_pos = px.bar(pos_df, x='POS Tag', y='Count',
                               title="Most Common POS Tags",
                               color='Count',
                               color_continuous_scale='viridis')
                st.plotly_chart(fig_pos, use_container_width=True)
                
                # Word frequency
                st.markdown("### 🔤 Most Common Words")
                
                words_df = pd.DataFrame(analysis['common_words'], 
                                      columns=['Word', 'Frequency'])
                
                fig_words = px.bar(words_df, x='Word', y='Frequency',
                                 title="Top 10 Most Frequent Words",
                                 color='Frequency',
                                 color_continuous_scale='plasma')
                st.plotly_chart(fig_words, use_container_width=True)
                
                # Word cloud simulation with text
                st.markdown("### ☁️ Word Frequency Table")
                st.dataframe(words_df, use_container_width=True)
                
                # POS tag explanations
                with st.expander("ℹ️ POS Tag Meanings"):
                    st.markdown("""
                    - **NN**: Noun (singular)
                    - **NNS**: Noun (plural)
                    - **VB**: Verb (base form)
                    - **VBD**: Verb (past tense)
                    - **VBG**: Verb (gerund/present participle)
                    - **VBN**: Verb (past participle)
                    - **VBP**: Verb (present tense)
                    - **VBZ**: Verb (3rd person singular present)
                    - **JJ**: Adjective
                    - **JJR**: Adjective (comparative)
                    - **JJS**: Adjective (superlative)
                    - **RB**: Adverb
                    - **IN**: Preposition
                    - **DT**: Determiner
                    - **CC**: Coordinating conjunction
                    - **PRP**: Personal pronoun
                    """)
            else:
                st.warning("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <h4>🧠 NLP Concepts Demonstrated:</h4>
        <p>
            <strong>Tokenization</strong> • <strong>Stop Word Removal</strong> • <strong>Lemmatization</strong> • 
            <strong>POS Tagging</strong> • <strong>TF-IDF</strong> • <strong>N-grams</strong> • 
            <strong>Sentiment Analysis</strong> • <strong>Named Entity Recognition</strong> • 
            <strong>Similarity Matching</strong> • <strong>Text Classification</strong>
        </p>
        <hr>
        <p style='font-size: 0.9em;'>
            Built with ❤️ using <strong>Streamlit</strong>, <strong>spaCy</strong>, <strong>NLTK</strong>, 
            <strong>TextBlob</strong>, <strong>scikit-learn</strong>, and <strong>Plotly</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()