import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import pickle
from streamlit_tags import st_tags
import altair as alt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from datetime import datetime, date
import time
import subprocess

nltk.download('punkt')
nltk.download('wordnet')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['indian_restaurant']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


st.set_page_config(
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
    <style>
            body {
            font-family: 'Comic Sans MS', sans-serif !important;
            background-color: #0096c7;  
            }
            section
            p{
                text-align: justify;
            }        
            h1, h2, h3, h4, h5, h6{
                text-align:center;
                
            }
            h1{
                color: #0096c7;
            }
            h2, h3, h4, h5, h6{
                margin: 20px 0px 5px 0px;
            }
            h2{
                color: #023e8a;
            }
            h3, h4, h5, h6{
                color: #0096c7;
            }
            .stImage image{
                margin-top: 50px;
                padding-top: 50px;
            }
            .stButton button{
                color: #0096c7;
                display: block;
                margin-left: auto;
                margin-right: auto;
                margin-top: 29px;
                border: 1.5px solid #0096c7;
                transition: transform 0.3s ease-in-out;
            }
            .stButton button:hover{
                transform: scale(1.1);
            }
            .body-text{
                margin: 10px;
                padding: 20px;
            }
            .overview{
                background-color: #ade8f4; 
                width: 50%;
                border-radius: 8px;
                padding: 20px;
                margin: auto;
                # margin: 10px 150px;
            }
            @media screen and (max-width: 992px) {
                .overview {
                    width:75%;
                }
            }
            @media screen and (max-width: 600px) {
                .overview {
                    width:90%;
                }
            }
            .overview-2{
                border: 2px solid #ade8f4; 
                border-radius: 8px;
                padding: 20px;
                margin: 10px 20px;
                text-align: center;
                font-size: 1.2rem;
            }
            .highlight{
                text-align: center;
                color: #03045e;
                font-size: 1.4rem;
                font-weight: bold;
            }
            .highlighted-container{
                width: 100px; /* Adjust the width */
                height: 50px;
                display: flex; /* Use Flexbox */
                align-items: center; /* Vertically center the content */
                justify-content: center; /* Horizontally center the content */
                margin: 30px auto; /* Set vertical margin and center horizontally */
                padding: 10px;
                border-radius: 8px;
                background-color: #ade8f4;
                transition: transform 0.3s ease-in-out, background-color 0.3s ease-in-out;
            }
            .highlighted-container:hover{
                transform: scale(1.1);
                background-color: #03045e;
            }
            .highlighted-container1{
                # width: 100px; /* Adjust the width */
                height: 300px;
                display: flex; /* Use Flexbox */
                align-items: center; /* Vertically center the content */
                justify-content: center; /* Horizontally center the content */
                margin: 30px auto; /* Set vertical margin and center horizontally */
                padding: 10px;
                border-radius: 8px;
                background-color: #03045e;
                transition: transform 0.3s ease-in-out, background-color 0.3s ease-in-out;
            }
            .highlighted-container1:hover{
                transform: scale(1.1);
                background-color: #03045e;
            }
            .pattern-container{
                width: 150px; /* Adjust the width */
                height: 50px;
                display: flex; /* Use Flexbox */
                align-items: center; /* Vertically center the content */
                justify-content: center; /* Horizontally center the content */
                margin: 30px auto; /* Set vertical margin and center horizontally */
                padding: 10px;
                border-radius: 8px;
                background-color: #ade8f4;
                transition: transform 0.3s ease-in-out, background-color 0.3s ease-in-out;
            }
            .pattern-container:hover{
                transform: scale(1.1);
                background-color: #03045e;
            }
            .highlighted-container3{
                color: black;
                text-align: center;
                margin: 30px 150px;
                padding: 10px;
                border-radius: 8px;
                background-color:#ade8f4;
                transition: transform 0.3s ease-in-out, background-color 0.3s ease-in-out;
            }
            .highlighted-container3:hover{
                transform: scale(1.1);
                background-color: #03045e;
            }
            # span{
            #     font-size: 1.4rem;
            #     color: #0096c7;
            # }
    </style>


""", unsafe_allow_html=True)


session_state = st.session_state
if 'page_index' not in session_state:
    session_state.page_index = 0

st.sidebar.markdown("<h1>Restaurant Chatbot</h1>", unsafe_allow_html=True)
section = st.sidebar.radio("INDEX ", ["Introduction", "Types of Chatbot", "Data Exploration - 1", "Data Exploration - 2", "How Does A chatbot Work?", "Lemmatization",
                           "Bag of Words - 1", "Bag of Words - 2", "Training", "Test Your Model!", "Chat with your bot"],  index=session_state.page_index)

if section == "Introduction":
    st.markdown('<div class="center"><h1>Restaurant Chatbot</h1></div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        image1 = Image.open('media/img1.png')
        st.image(image1, caption='')
    with col3:
        pass

    st.markdown(
        """
                    <div class="body-text"><p>Hey there, curious minds! Imagine having a digital friend who's always ready to chat, answer your questions, and help you out whenever you need. That's exactly what a chatbot does!

                    At its core, a chatbot is a clever computer program designed to understand your messages and respond in a way that makes sense. It's like having a conversation with a really smart machine!

                    How does it work, you ask? Well, these chatbots use something called Natural Language Processing (NLP), a fancy term for understanding human language. They analyze the words and sentences you type or say, figure out what you're asking or saying, and then come up with a helpful response.

                    They're everywhere, from helping in games and apps to answering questions on websites. And guess what? We're going to learn all about these amazing chatbots and maybe even build our own friendly digital pals. Get ready to dive into the awesome world of chatbots!</p> </div>
        """, unsafe_allow_html=True)

elif section == "Types of Chatbot":
    st.markdown('<div class="center"><h1>Types of Chatbots</h1></div>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        image1 = Image.open('media/img2.png')
        st.image(image1, caption='')
    with col3:
        pass

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            '<div class="center"><h3>Rule Based Chatbots</h3></div>', unsafe_allow_html=True)
        st.markdown("""<div class="body-text"><p>Imagine you have a list of specific rules, like a treasure map, telling you exactly what to do in different situations. A rule-based chatbot follows these rules like a map. When someone asks a question, it looks at this map and picks the answer that matches the rules. It's a bit like a game where you know exactly what moves to make because you've memorized the rules. But, if the question doesn't fit the rules you have, it's like trying to play a game that doesn't follow the rules you know. </p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="center"><h3>AI Based Chatbots</h3></div>', unsafe_allow_html=True)
        st.markdown("""<div class="body-text"><p>AI chatbot is a super cool robot that learns from talking to people. It's like having a friend who gets smarter every time they talk to someone new. This robot learns how to understand different questions and find the best answers by looking at all the conversations it has. So, the more it talks to people, the smarter it becomes in figuring out what someone is asking and how to help them! It's like a robot friend that learns and gets better with each chat it has! </p></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        image1 = Image.open('media/img3.png')
        st.image(image1, caption='')
    with col3:
        pass

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            '<div class="center"><h3>Retrieval-Based Chatbots</h3></div>', unsafe_allow_html=True)
        st.markdown("""<div class="body-text"><p>Think of a retrieval-based chatbot like a very organized library. When you ask a question, this chatbot looks for the best answer in its library of pre-written responses. It's a bit like finding a book in a library where each question has a specific book with the answer. If the question matches one of the books in the library, it gives you the answer from that book. But if your question is not in any of the books, it might not know how to help you. </p></div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="center"><h3>Generative Chatbots</h3></div>', unsafe_allow_html=True)
        st.markdown("""<div class="body-text"><p>Now, imagine a generative chatbot as a creative storyteller. Instead of picking pre-written answers, it makes up its own responses like telling a new story. This chatbot creates answers based on what it has learned from lots of conversations. It's like having a friend who doesn't just have books but can write new stories too. So, even if it hasn't heard your exact question before, it tries to make up an answer that might help, just like telling a new story every time someone asks a question! </p></div>""", unsafe_allow_html=True)


elif section == "Data Exploration - 1":
    st.markdown("""<div class="center"><h1>Let's Explore The Dataset</h1></div>""",
                unsafe_allow_html=True)
    st.markdown("""
                <div><p> The dataset used to train a chatbot is a crucial component that determines how well the chatbot understands and responds to user queries. This dataset contains various tags, associated patterns (user inputs), and corresponding responses that the chatbot can use to engage in conversations related to different topics or intents and we store this in a JSON file. </p></div>
                """, unsafe_allow_html=True)

    st.markdown("<ol>", unsafe_allow_html=True)
    st.markdown(
        "<li><b>Tag:</b> Each tag represents a category or intent of user queries.</li>", unsafe_allow_html=True)

    st.markdown(
        "<li><b>Patterns:</b> Patterns are examples of what users might say or ask related to that specific tag.</li>", unsafe_allow_html=True)

    st.markdown(
        "<li><b>Responses:</b> Responses are the appropriate replies or actions that the chatbot should provide for each pattern.</li>", unsafe_allow_html=True)

    if st.button("Visualize Data"):
        # Load intents from the JSON file
        with open('intents1.json', 'r') as file:
            data = json.load(file)
        # Extracting the intents
        intents = data["indian_restaurant"]

        # Creating lists for tags, patterns, and responses
        tags = []
        patterns = []
        responses = []

        # Extracting data from JSON
        for intent in intents:
            tags.append(intent["tag"])
            # Convert list of patterns to comma-separated string
            # Keep the list of patterns as is
            patterns.append(intent["patterns"])
            # Keep the list of responses as is
            responses.append(intent["responses"])

        # Creating a DataFrame from the extracted data
        df = pd.DataFrame({
            "Tag": tags,
            "Patterns": patterns,
            "Responses": responses
        })

        # Displaying the DataFrame as a table using Streamlit
        st.write(df, use_container_width=True)

elif section == "Data Exploration - 2":
    st.markdown("""<div class="center"><h1>Let's see how much you have learnt about the dataset! </h1></div>""",
                unsafe_allow_html=True)
    st.markdown("""<div class="center"><p> Here's a short exercise for you. Given here are some of the sentences and you need to categorize them into tags, patterns and responses. </p></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        # Load the image
        image = "media/cloud1.png"  # Replace with your image path

        # Add the image to Streamlit
        st.image(image, use_column_width=True)

        # Add text on top of the image using HTML and CSS
        st.markdown(
            """
            <style>
                .text-overlay {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-30%, -160%);
                    color: blue;
                    font-size: 24px;
                    font-weight: bold;
                    text-shadow: 2px 2px 4px #000000;
                    z-index: 1;
                }
            </style>
            <div class="text-overlay">We are open from 11am-8pm, Tuesday-Sunday.</div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        col1, col2 = st.columns(2)
        with col1:
            # Allow users to answer from the list of options
            answer1 = st.selectbox(
                'Answer 1:', [' ', 'tags', 'patterns', 'responses'], key='answer1')
        with col2:
            if answer1 == 'responses':
                st.image(Image.open('media/correct.png').resize((30, 30)),
                         use_column_width=False)
            else:
                st.image(Image.open('media/cross.png').resize((30, 30)),
                         use_column_width=False)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Load the image
        image = "media/cloud1.png"  # Replace with your image path

        # Add the image to Streamlit
        st.image(image, use_column_width=True)

        # Add text on top of the image using HTML and CSS
        st.markdown(
            """
            <style>
                .text-overlay1 {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-40%, -390%);
                    color: blue;
                    font-size: 24px;
                    font-weight: bold;
                    text-shadow: 2px 2px 4px #000000;
                    z-index: 1;
                }
            </style>
            <div class="text-overlay1">food of the day.</div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        col1, col2 = st.columns(2)
        with col1:
            # Allow users to answer from the list of options
            answer2 = st.selectbox(
                'Answer 2:', [' ', 'tags', 'patterns', 'responses'], key='answer2')
        with col2:
            if answer2 == 'patterns':
                st.image(Image.open('media/correct.png').resize((30, 30)),
                         use_column_width=False)
            else:
                st.image(Image.open('media/cross.png').resize((30, 30)),
                         use_column_width=False)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Load the image
        image = "media/cloud1.png"  # Replace with your image path

        # Add the image to Streamlit
        st.image(image, use_column_width=True)

        # Add text on top of the image using HTML and CSS
        st.markdown(
            """
            <style>
                .text-overlay2 {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-40%, -380%);
                    color: blue;
                    font-size: 24px;
                    font-weight: bold;
                    text-shadow: 2px 2px 4px #000000;
                    z-index: 1;
                }
            </style>
            <div class="text-overlay2">Payments.</div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        col1, col2 = st.columns(2)
        with col1:
            # Allow users to answer from the list of options
            answer3 = st.selectbox(
                'Answer 3:', [' ', 'tags', 'patterns', 'responses'], key='answer3')
        with col2:
            if answer3 == 'tags':
                st.image(Image.open('media/correct.png').resize((30, 30)),
                         use_column_width=False)
            else:
                st.image(Image.open('media/cross.png').resize((30, 30)),
                         use_column_width=False)

    st.markdown("""<div class="center"><h3>Let's make some additions to the dataset! </h3></div>""",
                unsafe_allow_html=True)
    st.markdown("""<div class="center"><p> Now that you're familiar with the dataset, you can add your creative thoughts to it and assist in training the model. </p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="center"><p> Select any tag from the list of options and add your questions as patterns and the corresponding answers as responses. </p></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        pass

    with col2:
        tag_input = st.selectbox(
            'TAGS:', ['greeting', 'time', 'payments', 'food', 'delivery', 'vegetarian', 'alcohol', 'location', 'bye'], key='tag_input', index=None, placeholder="Select tag",)
        pattern_input = st.text_input('PATTERNS:')
        response_input = st.text_input('RESPONSES:')

    with col3:
        pass
    # Load existing data from JSON
    with open('intents1.json', 'r') as file:
        data = json.load(file)

    # Initialize empty lists for tags, patterns, and responses
    tags = []
    patterns = []
    responses = []

    # Extracting the intents
    intents = data["indian_restaurant"]

    # Extract existing data into lists
    for intent in intents:
        tags.append(intent["tag"])
        patterns.append(intent["patterns"])
        responses.append(intent["responses"])

    if st.button("Add Data"):
        # Find the index of the selected tag in the dropdown list
        tag_index = ['greeting', 'time', 'payments', 'food',
                     'delivery', 'vegetarian', 'alcohol', 'location', 'bye'].index(tag_input)
        # tag_index = tag_index-1
        # Append new data to the corresponding tag's lists
        if tag_input.strip() != '' and pattern_input.strip() != '' and response_input.strip() != '':
            # If tag is already present, append data to its corresponding index
            if tag_index < len(tags):
                patterns[tag_index].extend(pattern_input.split(','))
                responses[tag_index].extend(response_input.split(','))
            else:
                tags.append(tag_input)
                patterns.append(pattern_input.split(','))
                responses.append(response_input.split(','))

        # Combine existing and new data
        intents = [
            {"tag": tag, "patterns": pats, "responses": resp}
            for tag, pats, resp in zip(tags, patterns, responses)
        ]

        # Update the data JSON file
        with open('intents1.json', 'w') as file:
            json.dump({"indian_restaurant": intents}, file, indent=4)


elif section == "How Does A chatbot Work?":
    st.markdown("""<div class="center"><h1>How Does A chatbot Work? </h1></div>""",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div class="center"><h3>Data Preparation</h3></div>', unsafe_allow_html=True)
        st.markdown("""<div class="body-text"><li>The data is stored in a JSON file containing intents, patterns (user messages), and responses.</li>
<li>We tokenize sentences into words and extracts these words along with their corresponding intents, preparing the data for further processing. </li></div>""", unsafe_allow_html=True)
        st.markdown(
            '<div class="center"><h3>Text Preprocessing</h3></div>', unsafe_allow_html=True)
        st.markdown("""<div class="body-text"><li><b>Lemmatization:</b> We convert words to their base or dictionary form. For instance, verbs in different tenses will be converted to their base form (e.g., "running" to "run").</li>
<li><b>Bag of Words (BoW):</b> It creates a numerical representation of text data. Each word in the vocabulary is assigned an index, and for each sentence, a vector is created with a 1 at the index of each word present and 0s elsewhere. This representation captures word presence but not the word order or context. </li></div>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="center"><h3>Prediction and Response</h3></div>', unsafe_allow_html=True)
    st.markdown("""<div class="body-text"><li>User input is processed by converting it into a BoW representation using the same preprocessing steps as during training.</li>
<li>The trained model predicts the intent class based on the BoW representation of the user input.</li>
<li>The code identifies intents with a probability higher than a defined threshold  and sorts them by probability.</li>
<li>Finally, a suitable response associated with the predicted intent is selected randomly from the intents JSON file and displayed as the chatbot's reply.</li></div>""", unsafe_allow_html=True)

    with col2:
        # Load the image
        image = "media/lemmatization.png"  # Replace with your image path
        # Add the image to Streamlit
        st.image(image, use_column_width=False)

        # Load the image
        image = "media/bow.png"  # Replace with your image path
        # Add the image to Streamlit
        st.image(image, use_column_width=True)
    # Load the image
    image = "media/proc.png"  # Replace with your image path

    # Add the image to Streamlit
    st.image(image, use_column_width=True, caption="Prediction and Response")

elif section == "Lemmatization":
    st.markdown("""<div class="center"><h1>Explore how dataset is processed! </h1></div>""",
                unsafe_allow_html=True)
    st.markdown(
        '<div class="center"><h3>Lemmatization</h3></div>', unsafe_allow_html=True)
    st.markdown("""<div class="body-text"><p>You know how words can sometimes change a bit when we talk about different things? Like 'walk' becomes 'walking' or 'walked'? Lemmatization is a bit like finding the main, original word. </p>
    <p>So, if we have 'running', 'ran', and 'run', lemmatization helps us figure out that they all come from the word 'run'. It's like finding the boss word that connects all these different versions together. </p>
    <p>It helps us understand words better and put them in order, making it easier for computers to understand too! Lemmatization is like finding the main word among different forms of the same word. </p>
    <p>Here is a small exercise for you to understand lemmatization. </p></div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """<div class="highlighted-container"><h5>walking</h5></div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class="highlighted-container"><h5>walked</h5></div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class="highlighted-container"><h5>walks</h5></div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class="highlighted-container"><h5>walk</h5></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="center"><h3>   </h3></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="center"><h3>   </h3></div>',
                    unsafe_allow_html=True)
        # Load the image
        image = "media/rightarrow.webp"  # Replace with your image path
        # Add the image to Streamlit
        st.image(image, use_column_width=True)
    with col3:
        st.markdown('<div class="center"><h3>        </h3></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="center"><h3>        </h3></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="center"><h3>        </h3></div>',
                    unsafe_allow_html=True)
        answer_input = st.text_input('Answer:')
    with col4:

        st.markdown('<div class="center"><h3>        </h3></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="center"><h3>        </h3></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="center"><h3>        </h3></div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            check_lemma_button = st.button(
                'Check', key='check_lemma_button')
        with col2:
            if check_lemma_button:
                if answer_input == 'walk':
                    st.image(Image.open(
                        'media/correct.png').resize((30, 30)), use_column_width=False)
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)


elif section == "Bag of Words - 1":
    st.markdown("""<div class="center"><h1>Explore how dataset is processed! </h1></div>""",
                unsafe_allow_html=True)
    st.markdown(
        '<div class="center"><h3>Bag Of Words</h3></div>', unsafe_allow_html=True)
    st.markdown("""<div class="body-text"><p>
    It creates a numerical representation of text data. Each word in the vocabulary is assigned an index, and for each sentence, a vector is created with a 1 at the index of each word present and 0s elsewhere. This representation captures word presence but not the word order or context.  </p>
    <p> Imagine you have a big box of words from a story. The bag of words is like counting how many times each word appears in that box.</p>
    <p> So, if the story has lots of 'cats' or 'dogs,' the bag of words helps us count how many times those words show up. It's like making a list of all the words and how many times we see each one. </p>
    <p> This helps us understand which words are super important in the story and what it's mostly about. It's a cool way for computers to learn about stories too! </p> </div>""", unsafe_allow_html=True)
    st.markdown("""<div class="body-text"><p> Let's learn through an example.</p>
    <p> Sentences:
    <ul>
    <li><b>Sentence 1:</b> "The cat chased the mouse."</li>
    <li><b>Sentence 2:</b>  "The dog barked loudly."</li>
    <li><b>Sentence 3:</b>  "The mouse ran away quickly." </li></p></div>""", unsafe_allow_html=True)
    st.markdown(
        '<div class="center"><h4>Fill the table and create BOW representation of sentences</h4></div>', unsafe_allow_html=True)
    # Sample sentences and corresponding correct answers
    data = {
        'Sentences': ['Sentence 1', 'Sentence 2', 'Sentence 3'],
        'the': [0, 0, 0],
        'cat': [0, 0, 0],
        'chased': [0, 0, 0],
        'mouse': [0, 0, 0],
        'dog': [0, 0, 0],
        'barked': [0, 0, 0],
        'loudly': [0, 0, 0],
        'ran': [0, 0, 0],
        'away': [0, 0, 0],
        'quickly': [0, 0, 0]
    }
    data1 = {
        'Sentences': ['Sentence 1', 'Sentence 2', 'Sentence 3'],
        'the': [1, 1, 1],
        'cat': [1, 0, 0],
        'chased': [1, 0, 0],
        'mouse': [1, 0, 1],
        'dog': [0, 1, 0],
        'barked': [0, 1, 0],
        'loudly': [0, 1, 0],
        'ran': [0, 0, 1],
        'away': [0, 0, 1],
        'quickly': [0, 0, 1]
    }
    # Create a DataFrame from the sample data
    df = pd.DataFrame(data)

    # Display the table in Streamlit
    edited_df = st.data_editor(
        df, disabled=["Sentences"], use_container_width=True)
    correct_answers = {
        (0, 0): 1,  # First row, first column
        (1, 0): 1,  # First row, second column
        (2, 0): 1,  # First row, third column
        (0, 1): 1,  # Second row, first column
        (1, 1): 0,  # Second row, second column
        (2, 1): 0,  # Second row, third column
        (0, 2): 1,  # Third row, first column
        (1, 2): 0,  # Third row, second column
        (2, 2): 0,   # Third row, third column
        (0, 3): 1,
        (1, 3): 0,
        (2, 3): 1,
        (0, 4): 0,
        (1, 4): 1,
        (2, 4): 0,
        (0, 5): 0,
        (1, 5): 1,
        (2, 5): 0,
        (0, 6): 0,
        (1, 6): 1,
        (2, 6): 0,
        (0, 7): 0,
        (1, 7): 0,
        (2, 7): 1,
        (0, 8): 0,
        (1, 8): 0,
        (2, 8): 1,
        (0, 9): 0,
        (1, 9): 0,
        (2, 9): 1
    }
    if st.button('Check Answers'):
        # Extract data from edited_df
        user_answers = {(i, j): edited_df.iloc[i, j+1]
                        for i, j in correct_answers.keys()}

        # Compare user answers with correct answers
        result = all(user_answers[key] == value for key,
                     value in correct_answers.items())

        # Display validation result
        if result == True:
            st.success('That is correct!', icon="✅")
        else:
            st.error('Wrong Answer! Try again.')

    st.markdown(
        '<div class="center"><h4>Here is a graphical representation of the table </h4></div>', unsafe_allow_html=True)

    if st.button('Visualize Graphs 📈'):
        df = pd.DataFrame(data1)
        # Melt the DataFrame to long format for plotting
        df_melted = df.melt(id_vars='Sentences',
                            var_name='Words', value_name='Count')

        # Create a column chart using Altair
        chart = alt.Chart(df_melted).mark_bar().encode(
            x='Sentences',
            y='Count',
            color='Words'
        ).properties(
            width=alt.Step(60)  # Adjust width if needed
        )
        # Create a column chart using Altair
        chart1 = alt.Chart(df_melted).mark_bar().encode(
            x='Words',
            y='Count',
            color='Sentences'
        ).properties(
            width=alt.Step(60)  # Adjust width if needed
        )

        # Display the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)
        st.altair_chart(chart1, use_container_width=True)

elif section == "Bag of Words - 2":
    st.markdown("""<div class="center"><h1>Explore how dataset is processed! </h1></div>""",
                unsafe_allow_html=True)
    st.markdown(
        '<div class="center"><h3>Bag Of Words</h3></div>', unsafe_allow_html=True)
    st.markdown("""<div class="body-text"><p> Let's perform a quick exercise.</p>
    <p> Sentences:
    <ul>
    <li>John likes to watch movies. Mary likes movies too.</li>
    <li>Mary also likes to watch football games. </li></p>
    <h4> Now you need to calculate frequencies of given words.</h4></div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1, 2, 1, 2])
    with col1:
        john = st.number_input('John:', min_value=0, format="%d")
        likes = st.number_input('likes:', min_value=0, format="%d")
        to = st.number_input('to:', min_value=0, format="%d")
        watch = st.number_input('watch:', min_value=0, format="%d")
        football = st.number_input('football:', min_value=0, format="%d")
    with col2:
        col1, col2 = st.columns(2)
        with col1:
            check_john_button = st.button('Check for "John"')
        with col2:
            if check_john_button:
                if john == 1:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            check_likes_button = st.button('Check for "likes"')
        with col2:
            if check_likes_button:
                if likes == 3:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            check_to_button = st.button('Check for "to"')
        with col2:
            if check_to_button:
                if to == 2:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            check_watch_button = st.button('Check for "watch"')
        with col2:
            if check_watch_button:
                if watch == 2:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            check_football_button = st.button('Check for "football"')
        with col2:
            if check_football_button:
                if football == 1:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)

    with col3:
        mary = st.number_input('Mary:', min_value=0, format="%d")
        movies = st.number_input('movies:', min_value=0, format="%d")
        too = st.number_input('too:', min_value=0, format="%d")
        also = st.number_input('also:', min_value=0, format="%d")
        games = st.number_input('games:', min_value=0, format="%d")

    with col4:
        col1, col2 = st.columns(2)
        with col1:
            check_mary_button = st.button('Check for "Mary"')
        with col2:
            if check_mary_button:
                if mary == 2:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            check_movies_button = st.button('Check for "movies"')
        with col2:
            if check_movies_button:
                if movies == 2:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            check_too_button = st.button('Check for "too"')
        with col2:
            if check_too_button:
                if too == 1:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            check_also_button = st.button('Check for "also"')
        with col2:
            if check_also_button:
                if also == 1:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            check_games_button = st.button('Check for "games"')
        with col2:
            if check_games_button:
                if games == 1:
                    image1 = Image.open(
                        'media/correct.png').resize((30, 30))
                    st.image(image1, caption='')
                    # st.Image(Image.open('media/correct.png').resize((30, 30)), use_column_width=False)
                else:
                    st.image(Image.open(
                        'media/cross.png').resize((30, 30)), use_column_width=False)

    st.markdown("""<div class="center"><h4>Now generate the graph to visualise more. </h4></div>""",
                unsafe_allow_html=True)
    if st.button("Generate Graph📈"):
        # Data for the chart
        words = ['John', 'likes', 'to', 'watch', 'movies',
                 'Mary', 'too', 'also', 'games', 'football']
        frequencies = [1, 3, 2, 2, 2, 2, 1, 1, 1, 1]
        data = pd.DataFrame({'Words': words, 'Frequencies': frequencies})

        # Create a base chart
        base = alt.Chart(data).encode(
            x='Words:N',
            y='Frequencies:Q'
        )
        # Create bars
        bars = base.mark_bar().encode(
            tooltip=['Words', 'Frequencies']
        )

        # Create text for values
        text = base.mark_text(
            align='center',
            baseline='middle',
            dy=-5  # Offset for text position
        ).encode(
            text='Frequencies:Q'
        )
        # Combine bars and text
        chart = bars + text
        # Show chart in Streamlit
        st.altair_chart(chart, use_container_width=True)

        # # Create a figure and a set of subplots
        # fig, ax = plt.subplots(figsize=(10, 6))
        # # Setting the background color of the figure
        # fig.patch.set_facecolor('black')
        # # Plotting the data as a column chart
        # ax.bar(words, frequencies, color='green')
        # # Setting chart background color
        # ax.set_facecolor('black')
        # # Setting labels and title
        # ax.set_xlabel('Words', color='blue')
        # ax.set_ylabel('Frequencies', color='blue')
        # # ax.set_title('Column Chart', color='blue')
        # ax.tick_params(axis='x', colors='white')
        # ax.tick_params(axis='y', colors='white')
        # # Show the chart
        # st.pyplot(fig)

    st.markdown("""<div class="center"><h4>List the top 5 most used words from the graph.</h4></div>""",
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        pass
    with col2:
        col1, col2 = st.columns(2)
        with col1:
            guess1 = st.selectbox(
                'Guess 1:', ['', 'John', 'likes', 'to', 'watch', 'movies', 'Mary', 'too', 'also', 'games', 'football'], key='guess1')
        with col2:
            if guess1 == 'likes':
                st.image(Image.open('media/correct.png').resize((30, 30)),
                         use_column_width=False)
            else:
                st.image(Image.open('media/cross.png').resize((30, 30)),
                         use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            guess2 = st.selectbox(
                'Guess 2:', ['', 'John', 'likes', 'to', 'watch', 'movies', 'Mary', 'too', 'also', 'games', 'football'], key='guess2')
        with col2:
            if guess2 == 'to' or guess2 == 'watch' or guess2 == 'movies' or guess2 == 'Mary':
                st.image(Image.open('media/correct.png').resize((30, 30)),
                         use_column_width=False)
            else:
                st.image(Image.open('media/cross.png').resize((30, 30)),
                         use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            guess3 = st.selectbox(
                'Guess 3:', ['', 'John', 'likes', 'to', 'watch', 'movies', 'Mary', 'too', 'also', 'games', 'football'], key='guess3')
        with col2:
            if guess3 == 'to' or guess3 == 'watch' or guess3 == 'movies' or guess3 == 'Mary':
                st.image(Image.open('media/correct.png').resize((30, 30)),
                         use_column_width=False)
            else:
                st.image(Image.open('media/cross.png').resize((30, 30)),
                         use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            guess4 = st.selectbox(
                'Guess 4:', ['', 'John', 'likes', 'to', 'watch', 'movies', 'Mary', 'too', 'also', 'games', 'football'], key='guess4')
        with col2:
            if guess4 == 'to' or guess4 == 'watch' or guess4 == 'movies' or guess4 == 'Mary':
                st.image(Image.open('media/correct.png').resize((30, 30)),
                         use_column_width=False)
            else:
                st.image(Image.open('media/cross.png').resize((30, 30)),
                         use_column_width=False)
        col1, col2 = st.columns(2)
        with col1:
            guess5 = st.selectbox(
                'Guess 5:', ['', 'John', 'likes', 'to', 'watch', 'movies', 'Mary', 'too', 'also', 'games', 'football'], key='guess5')
        with col2:
            if guess5 == 'to' or guess5 == 'watch' or guess5 == 'movies' or guess5 == 'Mary':
                st.image(Image.open('media/correct.png').resize((30, 30)),
                         use_column_width=False)
            else:
                st.image(Image.open('media/cross.png').resize((30, 30)),
                         use_column_width=False)
    with col3:
        pass


elif section == "Training":
    st.markdown("""<div class="center"><h1>Let's understand in detail how training happens!</h1></div>""",
                unsafe_allow_html=True)
    st.markdown("""<div class="body-text"><h4>From the dropdown menu, select any one category of intents:</h4> </div>""",
                unsafe_allow_html=True)
    # Load data from JSON file
    with open('intents1.json', 'r') as file:
        data = json.load(file)
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        selected_intent = st.selectbox(
            'Intent:', ['greeting', 'time', 'payments', 'food', 'delivery', 'vegetarian', 'alcohol', 'location', 'bye'], key='intent',  index=None, placeholder='Select one of the intents..')
    user_intent = selected_intent
    # st.write(user_intent)
    with col3:
        pass
    # Extract intents
    intents = data["indian_restaurant"]
    lemmatizer = WordNetLemmatizer()
    # Display patterns and responses for the selected intent
    for intent in intents:
        if intent['tag'] == selected_intent:
            st.markdown(
                f"""
            <div class="highlighted-container1">
                <h3>Patterns:</h3>
                <ul>
                    {''.join(f'<li>{pattern}</li>' for pattern in intent['patterns'])}
                </ul>
                <h3>Responses:</h3>
                <ul>
                    {''.join(f'<li>{response}</li>' for response in intent['responses'])}
                </ul>
            </div>
            """,
                unsafe_allow_html=True
            )
            st.markdown("""<div class="body-text"><h4>Now we tokenize and then lemmatize each of the patterns, and create bag of words representation of the patterns.</h4> </div>""",
                        unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                for pattern in intent['patterns']:
                    st.markdown(
                        f"""<div class="pattern-container"><h6>{pattern}</h6></div>""", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="center"><h3>   </h3></div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="center"><h3>   </h3></div>',
                            unsafe_allow_html=True)
                # Load the image
                image = "media/rightarrow.webp"  # Replace with your image path
                # Add the image to Streamlit
                st.image(image, use_column_width=True)

            with col3:
                words = []
                ignore_letters = ['?', '!', ',', '.']
                word_patterns = []
                pattern_labels = []
                for pattern in intent['patterns']:
                    # words_from_pattern.append(pattern.split())
                    pattern_labels.append(pattern)
                    word_list = nltk.word_tokenize(pattern)
                    words.extend(word_list)
                    word_patterns.append(word_list)
                words = [lemmatizer.lemmatize(word)
                         for word in words if word not in ignore_letters]
                words = sorted(set(words))
                keywords = st_tags(
                    label='# Enter Words:',
                    text='Press enter to add more',
                    value=[words[0]],
                    # suggestions=['five', 'six', 'seven', 'eight',
                    #              'nine', 'three', 'eleven', 'ten', 'four'],
                    maxtags=len(words),
                )
                keywords = sorted(set(keywords))
                # st.write(keywords)
                if st.button("Verify Words"):
                    if keywords == words:
                        st.success("Correct: All words match!")
                    else:
                        st.error(
                            f"Incorrect! The words are: {', '.join(words)}")
            bow = []
            if st.button("Generate BOWs"):
                for pattern in word_patterns:
                    # pattern_labels.append(pattern)
                    pattern_bow = [
                        1 if word in pattern else 0 for word in words]
                    bow.append(pattern_bow)
                data = {'Pattern': pattern_labels}
                for idx, word in enumerate(words):
                    data[word] = [bow_row[idx] for bow_row in bow]
                df = pd.DataFrame(data)
                st.dataframe(df)
            st.markdown("""<div class="body-text"><p>These BOWs along with the tag is fed to the model for training. The model learns about the data and whenever a similar question or pattern is asked it recognizes the tag. It randomly picks one response from all of the responses associated with the tag as the answer from chatbot.</p> </div>""",
                        unsafe_allow_html=True)

            def run_training():
                try:
                    process = subprocess.Popen(
                        ['python', 'train.py'], stdout=subprocess.PIPE, universal_newlines=True)
                    for line in process.stdout:
                        st.write(line)
                        if 'Epoch' in line:
                            progress = line.strip().split()[-1]
                            # st.write(progress)
                            numerator, denominator = map(
                                float, progress.split('/'))
                            progress_bar.progress(numerator / denominator)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    

            if st.button("Train Model"):
                st.write("Training in progress...")
                progress_bar = st.progress(0)
                run_training()
                # progress_bar.empty()
                st.success("Training completed!")

elif section == "Test Your Model!":
    st.markdown("""<div class="center"><h1>Test Your Model!</h1></div>""",
                unsafe_allow_html=True)

    question = st.text_input('Type a question to ask the chatbot!')
    intents = json.loads(open('intents1.json').read())

    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbotmodel.h5')
    lemmatizer = WordNetLemmatizer()

    qwords = clean_up_sentence(question)

    if st.button("Submit"):
        st.markdown(
            f"""
                <div class="highlighted-container1">
                    <h3>Words after tokenization and lemmatization:</h3>
                    <ul>
                        {''.join(f'<li>{pattern}</li>' for pattern in qwords)}
                    </ul>
                </div>
                """,
            unsafe_allow_html=True
        )
        st.markdown("""<div class="center"><h3>Bag of Words representation for the above question</h3></div>""",
                    unsafe_allow_html=True)
        bag_representation = bag_of_words(question)
        bag_data = {'Words': words, 'Question': bag_representation}
        bag_df = pd.DataFrame(bag_data)
        bag_df_transposed = bag_df.T
        # Display bag of words in a DataFrame
        st.dataframe(bag_df_transposed)
        st.markdown(
            """<div class="center"><h3>Submit BOW to model</h3></div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        with col2:
            image = "media/da.png"  # Replace with your image path
            # Add the image to Streamlit
            st.image(image, use_column_width=True)
        with col3:
            pass
        st.markdown("""<div class="center"><h3>What is the intent of the question?</h3></div>""",
                    unsafe_allow_html=True)

    if st.button("Predict"):
        answer = predict_class(question)
        st.markdown("""<div class="center"><h3>What is the intent of the question?</h3></div>""",
                    unsafe_allow_html=True)
        st.markdown(
            f"""<div class="pattern-container"><h5>{answer[0]['intent']}</h5></div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class="highlighted-container3"><h5>The intent with highest probability is chosen.</h5></div>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        with col2:
            image = "media/da.png"  # Replace with your image path
            # Add the image to Streamlit
            st.image(image, use_column_width=True)
        with col3:
            pass
        st.markdown("""<div class="center"><h4>The chatbot randomly chooses one of the responses associated with the recognized intent of the question as the answer.</h4></div>""",
                    unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("media/chat.webp", use_column_width=True)
        with col2:
            st.markdown(
                f"""<div class="highlighted-container3"><h5>{get_response(answer, intents)}</h5></div>""", unsafe_allow_html=True)

elif section == "Chat with your bot":
    intents = json.loads(open('intents1.json').read())
    lemmatizer = WordNetLemmatizer()

    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbotmodel.h5')

    st.markdown("""<div class="center"><h1>Try the new Chatbot!!</h1></div>""",
                unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type something!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            ints = predict_class(prompt)
            res = get_response(ints, intents)
            assistant_response = res
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
