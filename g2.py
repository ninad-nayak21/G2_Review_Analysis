# import requests
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from transformers import pipeline
# url = "https://data.g2.com/api/v1/survey-responses"
# headers = {
#     "Authorization": "Token token=8b890ca6886d1036319d19129a29031dd0c865bd2623b01ae6ae7c5e6941e456",
#     "Content-Type": "application/vnd.api+json"
# }

# response = requests.get(url, headers=headers)
# like = []
# dislike = []
# benefit = []
# recomendation = []
# secondary = {}

# if response.status_code == 200:
#     data = response.json()
#     survey_responses = data["data"]

#     for survey_response in survey_responses:
#         attributes = survey_response["attributes"]
#         comment_answers = attributes["comment_answers"]
#         secondary_answers = attributes["secondary_answers"]

#         # Comment Answers
#         for answer_type, answer_data in comment_answers.items():
#             if comment_answers[answer_type]['text'] == "What do you like best about the product?":
#                 like.append(comment_answers[answer_type]['value'])
#             if comment_answers[answer_type]['text'] == "What do you dislike about the product?":
#                 dislike.append(comment_answers[answer_type]['value'])
#             if comment_answers[answer_type]['text'] == "What problems is the product solving and how is that benefiting you?":
#                 benefit.append(comment_answers[answer_type]['value'])
#             if comment_answers[answer_type]['text'] == "Recommendations to others considering the product:":
#                 recomendation.append(comment_answers[answer_type]['value'])

#         # Secondary Answers
#         for answer_type, answer_data in secondary_answers.items():
#             text = secondary_answers[answer_type]['text']
#             value = secondary_answers[answer_type]['value']
#             if text in secondary:
#                 secondary[text].append(value)
#             else:
#                 secondary[text] = [value]

#     # Calculate averages for secondary answers
#     for key, values in secondary.items():
#         average = sum(values) / len(values)
#         secondary[key] = average

# else:
#     print("Failed to retrieve data. Status code:", response.status_code)

# def split_sentences(sentence_list):
#     splitted_list = []
#     for sentence in sentence_list:
#         splitted_list.extend(sentence.split("."))
#     # Remove empty strings
#     splitted_list = [s.strip() for s in splitted_list if s.strip()]
#     return splitted_list

# # Split sentences in each list
# like_best_list = split_sentences(like)
# dislike_list = split_sentences(dislike)
# recommendation_list = split_sentences(recomendation)
# benefit_list = split_sentences(benefit)
# like_best_list = like_best_list + benefit_list + recommendation_list
# from sentence_transformers import SentenceTransformer
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # Function to preprocess text
# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove punctuation and special characters
#     text = re.sub(r'[^\w\s]', '', text)
#     # Tokenize the text
#     tokens = word_tokenize(text)
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stopwords.words('english')]
#     return ' '.join(tokens)

# # Load pre-trained BERT model
# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# user_input = input("Enter your question: ")
# user_input = preprocess_text(user_input)

# Positive and negative questions
questions_positive = [
    "What are the features?",
    "What are the advantages?",
    "How is the performance?",
    "What problems is the product solving and how is that benefiting you?",
    "How easy is it to use the product?"
]

questions_negative = [
    "What are the disadvantages?",
    "What are problems faced?",
    "How reliable is the product in terms of performance and stability?",
    "Are there any minor inconveniences or annoyances users might encounter?",
]

# # Preprocess positive and negative questions
# questions_positive_preprocessed = [preprocess_text(q) for q in questions_positive]
# questions_negative_preprocessed = [preprocess_text(q) for q in questions_negative]

# # Combine all questions
# all_questions = questions_positive_preprocessed + questions_negative_preprocessed

# # Encode questions using BERT
# question_embeddings = model.encode(all_questions)

# # Encode user input using BERT
# user_input_embedding = model.encode([user_input])

# # Calculate cosine similarity between user input and positive questions
# positive_similarity = cosine_similarity(user_input_embedding, question_embeddings[:len(questions_positive_preprocessed)])[0]
# # Calculate cosine similarity between user input and negative questions
# negative_similarity = cosine_similarity(user_input_embedding, question_embeddings[len(questions_positive_preprocessed):])[0]

# # Print the similarity scores
# print("Positive similarity scores:", positive_similarity)
# print("Negative similarity scores:", negative_similarity)
# cat = ""
# question = ""
# # Determine the category
# if max(positive_similarity) > max(negative_similarity):
#     cat = "positive"
#     print("Category: Positive")
#     print("Score:", max(positive_similarity))
#     # Find the index of the most similar question
#     most_similar_index = positive_similarity.argmax()
#     print("Most similar question (Positive):", questions_positive[most_similar_index])
#     print("Similarity score:", positive_similarity[most_similar_index])
#     question = questions_positive[most_similar_index]
# elif max(negative_similarity) > max(positive_similarity):
#     print("Category: Negative")
#     print("Score:", max(negative_similarity))
#     # Find the index of the most similar question
#     most_similar_index = negative_similarity.argmax()
#     print("Most similar question (Negative):", questions_negative[most_similar_index])
#     print("Similarity score:", negative_similarity[most_similar_index])
#     question = questions_negative[most_similar_index]
# else:
#     print("Category: Neutral")
# print(cat,question)
# qa_pipeline = pipeline("question-answering")
# answer = []

# if cat == "positive":
#   for i, context in enumerate(like_best_list):
#     # print(f"Context {i+1}:")
#     # print(context)
#     # print("Answer:", qa_pipeline(question=question, context=context)["answer"])
#     answer.append(qa_pipeline(question=question, context=context)["answer"])

# elif cat == "negative":
#   for i, context in enumerate(dislike_list):
#     # print(f"Context {i+1}:")
#     # print(context)
#     # print("Answer:", qa_pipeline(question=question, context=context)["answer"])
#     answer.append(qa_pipeline(question=question, context=context)["answer"])

# # print(answer)
    
# sorted_sentences = sorted(answer, key=lambda x: len(x))
# print(len(sorted_sentences))
# # Print the sorted sentences
# for sentence in sorted_sentences[-4:]:
#     print(sentence)
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import nltk
nltk.download('punkt')
# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)
questions_positive_preprocessed = [preprocess_text(q) for q in questions_positive]
questions_negative_preprocessed = [preprocess_text(q) for q in questions_negative]

# Load pre-trained BERT model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Function to split sentences
def split_sentences(sentence_list):
    splitted_list = []
    for sentence in sentence_list:
        splitted_list.extend(sentence.split("."))
    # Remove empty strings
    splitted_list = [s.strip() for s in splitted_list if s.strip()]
    return splitted_list

# Function to retrieve data
def retrieve_data():
    url = "https://data.g2.com/api/v1/survey-responses"
    headers = {
        "Authorization": "Token token=8b890ca6886d1036319d19129a29031dd0c865bd2623b01ae6ae7c5e6941e456",
        "Content-Type": "application/vnd.api+json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data["data"]
    else:
        st.error("Failed to retrieve data. Status code: {}".format(response.status_code))
        return None

# Function to process survey responses
def process_survey_responses(survey_responses):
    like = []
    dislike = []
    benefit = []
    recommendation = []
    secondary = {}

    for survey_response in survey_responses:
        attributes = survey_response["attributes"]
        comment_answers = attributes["comment_answers"]
        secondary_answers = attributes["secondary_answers"]

        # Comment Answers
        for answer_type, answer_data in comment_answers.items():
            if comment_answers[answer_type]['text'] == "What do you like best about the product?":
                like.append(comment_answers[answer_type]['value'])
            if comment_answers[answer_type]['text'] == "What do you dislike about the product?":
                dislike.append(comment_answers[answer_type]['value'])
            if comment_answers[answer_type]['text'] == "What problems is the product solving and how is that benefiting you?":
                benefit.append(comment_answers[answer_type]['value'])
            if comment_answers[answer_type]['text'] == "Recommendations to others considering the product:":
                recommendation.append(comment_answers[answer_type]['value'])

        # Secondary Answers
        for answer_type, answer_data in secondary_answers.items():
            text = secondary_answers[answer_type]['text']
            value = secondary_answers[answer_type]['value']
            if text in secondary:
                secondary[text].append(value)
            else:
                secondary[text] = [value]

    # Calculate averages for secondary answers
    for key, values in secondary.items():
        average = sum(values) / len(values)
        secondary[key] = average

    return like, dislike, benefit, recommendation, secondary

# Function to process user input
def process_user_input(user_input, model, questions_positive_preprocessed, questions_negative_preprocessed):
    # Encode questions using BERT
    question_embeddings = model.encode(questions_positive_preprocessed + questions_negative_preprocessed)

    # Encode user input using BERT
    user_input_embedding = model.encode([user_input])

    # Calculate cosine similarity between user input and positive questions
    positive_similarity = cosine_similarity(user_input_embedding, question_embeddings[:len(questions_positive_preprocessed)])[0]
    # Calculate cosine similarity between user input and negative questions
    negative_similarity = cosine_similarity(user_input_embedding, question_embeddings[len(questions_positive_preprocessed):])[0]

    # Determine the category
    if max(positive_similarity) > max(negative_similarity):
        cat = "positive"
        most_similar_index = positive_similarity.argmax()
        question = questions_positive[most_similar_index]
    elif max(negative_similarity) > max(positive_similarity):
        cat = "negative"
        most_similar_index = negative_similarity.argmax()
        question = questions_negative[most_similar_index]
    else:
        cat = "neutral"
        question = None

    return cat, question

# Function to find answers
def find_answers(cat, question, like_best_list, dislike_list, qa_pipeline):
    answers = []
    if cat == "positive":
        for context in like_best_list:
            answers.append(qa_pipeline(question=question, context=context)["answer"])
    elif cat == "negative":
        for context in dislike_list:
            answers.append(qa_pipeline(question=question, context=context)["answer"])
    return sorted(answers, key=lambda x: len(x))[-4:]

def display_image(image_url):
    if image_url:
        st.image(image_url, caption='User Image', use_column_width=True)
    else:
        st.write("No image available")

def main():
    st.title("Product Feedback Analysis")

    user_input = st.text_input("Enter your question:")

    if st.button("Submit"):
        user_input = preprocess_text(user_input)

        data = retrieve_data()

        if data:
            survey_responses = process_survey_responses(data)

            like_best_list = split_sentences(survey_responses[0] + survey_responses[2] + survey_responses[3])
            dislike_list = split_sentences(survey_responses[1])
            recommendation_list = split_sentences(survey_responses[4])
            like_best_list = like_best_list + recommendation_list

            cat, question = process_user_input(user_input, model, questions_positive_preprocessed, questions_negative_preprocessed)

            if question:
                qa_pipeline = pipeline("question-answering")
                answers = find_answers(cat, question, like_best_list, dislike_list, qa_pipeline)

                st.subheader("Top Answers")
                for answer in answers:
                    
                    # print(data)
                    # print(len(data["data"]))
                    # print(data["data"])
                    for survey_response in data:
                        attributes = survey_response["attributes"]
                        # print(survey_response["attributes"])
                        # print(attributes['comment_answers'])
                        comment_answers = attributes["comment_answers"]
                        for answer_type,answer_data in comment_answers.items():
                            # print(comment_answers[answer_type]['value'])
                            if answer in comment_answers[answer_type]['value']:
                                st.write("-"*20)
                                secondary_answers = attributes["secondary_answers"]
                                star_rating = attributes["star_rating"]
                                votes_up = attributes["votes_up"]
                                votes_down = attributes["votes_down"]
                                username = attributes["user_name"]
                                image_url = attributes["user_image_url"]
                                country_name = attributes["country_name"]
                                submitted_date = attributes['submitted_at']
                                updated_id = attributes["updated_at"]
                                title = attributes["title"]
                                # Display user information
                                st.write(f"Username: {username}")
                                st.write(f"Country: {country_name}")
                                # display_image(image_url) 
                                st.write(f"Submitted Date: {submitted_date}")
                                st.write(f"Updated Date: {updated_id}")
                                st.write(f"Title: {title}")
                                highlight_text = answer
                                original_sentence = comment_answers[answer_type]['value']
                                colored_text = f"<span style='color: black; background-color: white;'>**{highlight_text}**</span>"
                                formatted_sentence = original_sentence.replace(highlight_text, colored_text)
                                # st.write(formatted_sentence, unsafe_allow_html=True)

                                st.markdown(formatted_sentence, unsafe_allow_html=True)
                                # Display secondary answers
                                st.write("Secondary Answers:")
                                for answer_type, answer_data in secondary_answers.items():
                                    st.write(f"- {secondary_answers[answer_type]['text']}: {secondary_answers[answer_type]['value']}")
                                
                                # Display star rating, votes, etc.
                                st.write(f"Star Rating: {star_rating}")
                                st.write(f"Votes Up: {votes_up}")
                                st.write(f"Votes Down: {votes_down}")
                                st.write(answer)
                                st.write("-"*20)
                                break
                    

if __name__ == "__main__":
    main()
