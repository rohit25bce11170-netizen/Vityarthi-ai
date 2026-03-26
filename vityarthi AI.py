import streamlit as st
import nltk
import random
import json
import datetime
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')


faq_data = {
    "admission": [
        "Admissions are open from June to August.",
        "You can apply online through our official website.",
        "Documents required include 10th and 12th marksheets."
    ],
    "courses": [
        "We offer BCA, BBA, MBA, and BSc programs.",
        "Each course duration is 3 years except MBA which is 2 years."
    ],
    "fees": [
        "The fee for BCA is 50000 per year.",
        "MBA fee is approximately 120000 per year.",
        "Scholarships are available for meritorious students."
    ],
    "timing": [
        "College timing is from 9 AM to 4 PM.",
        "Office hours are from 10 AM to 5 PM."
    ],
    "hostel": [
        "Hostel facility is available for boys and girls.",
        "Hostel includes food and WiFi."
    ],
    "contact": [
        "You can contact us at 1234567890.",
        "Email us at info@college.com."
    ]
}


corpus = []
for key in faq_data:
    corpus.extend(faq_data[key])

greetings = ["hello", "hi", "hey"]
greet_responses = ["Hello! How can I assist you?", "Hi! Ask me anything.", "Hey there!"]


def greet(text):
    for word in text.lower().split():
        if word in greetings:
            return random.choice(greet_responses)



def get_response(user_input):
    user_input = user_input.lower()

    all_sentences = corpus + [user_input]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_sentences)

    similarity = cosine_similarity(vectors[-1], vectors)
    idx = similarity.argsort()[0][-2]

    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]

    if score < 0.2:
        return "Sorry, I couldn't understand. Please rephrase your question."

    return corpus[idx]



def save_chat(user, bot):
    chat = {
        "time": str(datetime.datetime.now()),
        "user": user,
        "bot": bot
    }

    try:
        with open("chat_history.json", "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(chat)

    with open("chat_history.json", "w") as f:
        json.dump(data, f, indent=4)



st.set_page_config(page_title="Smart AI Chatbot", layout="centered")

st.title("Smart AI Customer Support Chatbot")
st.write("Ask your queries about college services")


if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Enter your message:")

if user_input:

    response = greet(user_input)

    if not response:
        response = get_response(user_input)


    save_chat(user_input, response)


    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response))

for sender, msg in st.session_state.messages:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")

if st.button("Show Chat History"):
    try:
        with open("chat_history.json", "r") as f:
            history = json.load(f)
            for chat in history[-5:]:
                st.write(chat)
    except:
        st.write("No history found.")
