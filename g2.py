import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

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

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

questions_positive_preprocessed = [preprocess_text(q) for q in questions_positive]
questions_negative_preprocessed = [preprocess_text(q) for q in questions_negative]

def split_sentences(sentence_list):
    splitted_list = []
    for sentence in sentence_list:
        splitted_list.extend(sentence.split("."))
    splitted_list = [s.strip() for s in splitted_list if s.strip()]
    return splitted_list

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

        for answer_type, answer_data in comment_answers.items():
            if comment_answers[answer_type]['text'] == "What do you like best about the product?":
                like.append(comment_answers[answer_type]['value'])
            if comment_answers[answer_type]['text'] == "What do you dislike about the product?":
                dislike.append(comment_answers[answer_type]['value'])
            if comment_answers[answer_type]['text'] == "What problems is the product solving and how is that benefiting you?":
                benefit.append(comment_answers[answer_type]['value'])
            if comment_answers[answer_type]['text'] == "Recommendations to others considering the product:":
                recommendation.append(comment_answers[answer_type]['value'])

        for answer_type, answer_data in secondary_answers.items():
            text = secondary_answers[answer_type]['text']
            value = secondary_answers[answer_type]['value']
            if text in secondary:
                secondary[text].append(value)
            else:
                secondary[text] = [value]

    for key, values in secondary.items():
        average = sum(values) / len(values)
        secondary[key] = average

    return like, dislike, benefit, recommendation, secondary

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def process_user_input(user_input, model, questions_positive_preprocessed, questions_negative_preprocessed):
    question_embeddings = model.encode(questions_positive_preprocessed + questions_negative_preprocessed)
    user_input_embedding = model.encode([user_input])

    positive_similarity = cosine_similarity(user_input_embedding, question_embeddings[:len(questions_positive_preprocessed)])[0]
    negative_similarity = cosine_similarity(user_input_embedding, question_embeddings[len(questions_positive_preprocessed):])[0]

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
    st.title("G2 Review Analysis")

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
                                country_name = attributes["country_name"]
                                submitted_date = attributes['submitted_at']
                                updated_id = attributes["updated_at"]
                                title = attributes["title"]
                                # Display user information
                                st.write(f"Username: {username}")
                                st.write(f"Country: {country_name}")
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
