from recommendation_engine import recommendation_engine, Candidate, selection, generate_candidates
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import json
import time
import uuid
import os
import requests
from sentence_transformers import SentenceTransformer, util
import numpy as np
import traceback
import torch
import re

# http://127.0.0.1:8000/docs#/ interactive API documentation
# http://127.0.0.1:8000/redoc  ReDoc interactive API documentation
# http://127.0.0.1:8000/openapi.json OpenAPI json scheme

# Load the .env file
load_dotenv()

# Set up OpenAI API credentials
openai.api_key = os.getenv('OPENAI_API_KEY')

# File to store errors
ERRORS_FILE = "errors.json"

# File to store responses
RESPONSE_FILE = "response.json"

# Setup FastAPI
app = FastAPI()

# Default parameters class
class DefaultParams:
    def __init__(self):
        self.gpt_model = "gpt-3.5-turbo"
        self.temperature = 0
        self.max_tokens = 250
        self.top_p = 0.6
        self.frequency_penalty = 0.6
        self.presence_penalty = 0
        self.bert_model = 'all-MiniLM-L6-v2'
        self.z_score_threshold = 3
        self.cutoff_skills = 0.5
        self.number_of_output_skills = 3
        self.missing_skills_cutoff = 0.4
        self.soft_skills_number = 8
        self.tech_skills_number = 5


    # Return list of paramaters
    def get_dict(self):
        return {
            "gpt_model": self.gpt_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "bert_model": self.bert_model,
            "z_score_threshold": self.z_score_threshold,
            "skills_cutoff": self.cutoff_skills,
            "number_of_output_skills": self.number_of_output_skills,
            "missing_skills_cutoff": self.missing_skills_cutoff,
            "soft_skills_number": self.soft_skills_number,
            "tech_skills_number": self.tech_skills_number

        }

# Default parameters instance
default_params = DefaultParams()

# Set up SentenceTransformer model
nlp_model = SentenceTransformer(default_params.bert_model)

# API parameters class
class ApiParams(BaseModel):
    gpt_model: str = default_params.gpt_model
    temperature: float = default_params.temperature
    max_tokens: int = default_params.max_tokens
    top_p: float = default_params.top_p
    frequency_penalty: float = default_params.frequency_penalty
    presence_penalty: float = default_params.presence_penalty
    nlp_model: str = default_params.bert_model
    z_score_threshold: float = default_params.z_score_threshold
    cutoff_skills: float = default_params.cutoff_skills
    number_of_output_skills: int = default_params.number_of_output_skills
    missing_skills_cutoff: float = default_params.missing_skills_cutoff
    soft_skills_number: int = default_params.soft_skills_number
    tech_skills_number: int = default_params.tech_skills_number

class FeedbackInfo(BaseModel):
    user_id: str
    message: str
    gpt_first_response: str
    gpt_second_response: str
    skills: list
    user_vector: list
    total_time: float
    assign_skills_time: float
    total_price: float
    total_tokens: float

class Message(BaseModel):
    message: str

# Generate user ID
def generate_user_id():
    return str(uuid.uuid4())

# Function to get current timestamp
def timestamp():
    try:
        # Get the current timestamp
        timestamp = time.time()

        # Convert timestamp to a datetime object
        datetime_obj = time.gmtime(timestamp)

        # Format datetime to show date and hour
        formatted_datetime = time.strftime("%Y-%m-%d %H:%M:%S", datetime_obj)

        return formatted_datetime

    except Exception as e:
        save_error(str(e), "timestamp", traceback.format_exc())
        return

def remove_html_tags(html_string):
    CLEANR = re.compile('<.*?>')
    cleantext = re.sub(CLEANR, '', html_string)
    return cleantext

# Function to handle and save errors
def save_error(error_message=str, function=str, traceback_info=str):
    error_data = {"user_id": FeedbackInfo.user_id,
                  "date": timestamp(),
                  "user input": FeedbackInfo.message,
                  "gpt_response": FeedbackInfo.gpt_first_response,
                  "gpt_feedback": FeedbackInfo.gpt_second_response,
                  "function": function,
                  "error_message": error_message,
                  "traceback": traceback_info
                  }

    if os.path.exists(ERRORS_FILE):
        with open(ERRORS_FILE, "r") as file:
            errors = json.load(file)
            errors.append(error_data)
        with open(ERRORS_FILE, "w") as file:
            json.dump(errors, file, indent=4)
    else:
        with open(ERRORS_FILE, "w") as file:
            json.dump([error_data], file, indent=4)

# Save responses from the program to json dict
def save_response():
    try:
        response = {
            "user_id": FeedbackInfo.user_id,
            "date": timestamp(),
            "user_input": remove_html_tags(FeedbackInfo.message),
            "gpt_response": FeedbackInfo.gpt_first_response,
            "gpt_feedback": FeedbackInfo.gpt_second_response,
            "feedback_skills": FeedbackInfo.skills,
            "user_vector": FeedbackInfo.user_vector,
            "total_time": round(FeedbackInfo.total_time, 2),
            "time_skills_assign": round(FeedbackInfo.assign_skills_time, 2),
            "total_price": FeedbackInfo.total_price,
            "total_tokens": FeedbackInfo.total_tokens,
            "parameters": default_params.get_dict()
        }

        if os.path.exists(RESPONSE_FILE):
            with open(RESPONSE_FILE, "r") as file:
                responses = json.load(file)
                responses.append(response)
            with open(RESPONSE_FILE, "w") as file:
                json.dump(responses, file, indent=4)
        else:
            with open(RESPONSE_FILE, "w") as file:
                json.dump([response], file, indent=4)

    except Exception as e:
        save_error(str(e), "save_response", traceback.format_exc())

def z_score_filtering(assigned_skills):
    try:
        # Z score filtering
        if len(assigned_skills) > 1:
            # Extract the numeric values into a separate list
            numeric_values = [item[0] for item in assigned_skills]

            # Calculate the mean and standard deviation
            mean = np.mean(numeric_values)
            std_dev = np.std(numeric_values)

            # Set the threshold for z-score (e.g., 2)
            z_score_threshold = default_params.z_score_threshold

            # Filter out values with z-scores greater than the threshold
            filtered_list = [item for item in assigned_skills if (item[0] - mean) / std_dev >= z_score_threshold]

            return filtered_list

        else:
            return assigned_skills

    except Exception as e:
        save_error(str(e), "z_score_filtering", traceback.format_exc())
        return assigned_skills

def assign_skills(sentences, inn_skills):
    try:
        # Encode the text
        embeddings_inn = nlp_model.encode(inn_skills)

        # Get the top 3 most similar sentences
        gpt_skills = []
        user_vector = []
        top_k = min(5, len(sentences))
        seen_indices_user_vector = set()  # To keep track of seen indices
        seen_indices_gpt_skills = set()

        for k in sentences:
            sentence_embedding = nlp_model.encode(k, convert_to_tensor=True)

            # Compute cosine similarity between all pairs
            cos_scores = util.cos_sim(sentence_embedding, embeddings_inn)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            print("")
            print("")
            print("Query:", k)

            first_skill = 0

            for score, idx in zip(top_results[0], top_results[1]):
                #print(inn_skills[idx], "(Score: {:.4f})".format(score))

                # Save the unique skills in user vector
                if idx.item() not in seen_indices_user_vector:
                    seen_indices_user_vector.add(idx.item())
                    user_vector.append([round(score.item(), 2), inn_skills[idx]])

                if first_skill == 0:
                    if idx.item() not in seen_indices_gpt_skills:
                        seen_indices_gpt_skills.add(idx.item())
                        gpt_skills.append([round(score.item(), 2), inn_skills[idx]])
                        first_skill = 1

            print("gpt_skills:", gpt_skills)




    except Exception as e:
        save_error(str(e), traceback.format_exc(), "assign_skills")
        return

    # Filter out skills with low scores
    filtered_skills = [
        [score, skill] for score, skill in gpt_skills if score >= default_params.cutoff_skills
    ]

    return filtered_skills, user_vector


def assign_skills_simple(sentences, inn_skills):
    try:

        user_vector = []
        top_k = min(5, len(sentences))

        # Encode the text
        embeddings_inn = nlp_model.encode(inn_skills)
        embeddings_sentences = nlp_model.encode(sentences)

        # Compute cosine similarity between all pairs
        cos_scores = util.cos_sim(embeddings_sentences, embeddings_inn)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            #print(inn_skills[idx], "(Score: {:.4f})".format(score))
            user_vector.append([round(score.item(), 2), inn_skills[idx]])



        # Filter out skills with low scores
        filtered_skills = [
            [score, skill] for score, skill in user_vector if score >= default_params.cutoff_skills
            ]



    except Exception as e:
        save_error(str(e), traceback.format_exc(), "assign_skills")
        return

    return filtered_skills, user_vector


def return_skills_improved(user_input, json_validation, inn_list):
    try:

        if json_validation is True:

            response_dict = json.loads(user_input[user_input.find('{'):])
            # printresponse_dict)

            sentences = []
            for skill, des in response_dict.items():
                sentences.append(des)

            skills, user_vector = assign_skills(sentences, inn_list)
            # print"Skills:",skills)

            #sorted_tags = sorted(skills, key=lambda x: x[0], reverse=True)
            print("Sklls:", skills)
            print("User Vector:", user_vector)

        else:
            inn_list = innential_skills()
            skills, user_vector = assign_skills_simple(user_input, inn_list)
            print("Sklls:", skills)
            print("User Vector:", user_vector)

    except Exception as e:
        save_error(str(e), traceback.format_exc(), "return_skills_improved")
        return

    # Assign only the top N skills to the output
    user_vector = sorted(user_vector, key=lambda x: x[0], reverse=True)
    user_vector = user_vector[:10]

    return skills, user_vector

# Return skills for the uer input
def return_skills(user_input):
    skills_list = innential_skills()

    response_dict = json.loads(user_input[user_input.find('{'):])
    try:
        try:
            tags = []
            for skill, des in response_dict.items():
                text = skill + " " + des
                skills = assign_skills(text, skills_list)
                # printskills)
                for score, skill in skills:

                    # Check if the skill is already in the tags list
                    skill_exists = False
                    for tag in tags:
                        if tag[1] == skill:
                            # If the skill is already in the list, add the score value to the existing one
                            tag[0] += score
                            skill_exists = True
                            break

                    # If the skill is not in the list, append a new entry
                    if not skill_exists:
                        tags.append([score, skill])

        except:
            # Extract skills from the JSON data
            tags = []
            for skill_info in response_dict["Skills"]:
                text = skill_info["Skill"] + " " + skill_info["Description"]
                skills = assign_skills(text, skills_list)
                for score, skill in skills:

                    # Check if the skill is already in the tags list
                    skill_exists = False
                    for tag in tags:
                        if tag[1] == skill:
                            # If the skill is already in the list, add the score value to the existing one
                            tag[0] += score
                            skill_exists = True
                            break

                # If the skill is not in the list, append a new entry
                if not skill_exists:
                    tags.append([score, skill])


    except Exception as e:
        save_error(str(e), traceback.format_exc(), "return_skills")
        return

    # Sort the tags list by the score value in descending order
    sorted_tags = sorted(tags, key=lambda x: x[0], reverse=True)

    # Round weights to 2 decimal places
    for tag in sorted_tags:
        tag[0] = round(tag[0], 2)

    return sorted_tags

# Get innential skills from API
def innential_skills():
    try:
        skills = requests.get("https://api.innential.com/scraper/skills-categories/public")
        skills.raise_for_status()
        json_data = skills.json()
        # print"Innential Skills API: successful")

    except requests.exceptions.RequestException as e:
        save_error(str(e), "innential_skills", traceback.format_exc())
        raise HTTPException(status_code=500, detail="An error occurred and has been saved.")
        return []

    except ValueError as e:
        save_error(str(e), "innential_skills", traceback.format_exc())
        return []

    subcategories = [list(subcategory_dict.keys())[0] for subcategory_list in json_data.values() for subcategory_dict in
                     subcategory_list]

    return subcategories

def validate_json(text):
    try:
        # Sometime GPT doesn't end with "}"
        last_brace_index = text.rfind("}")

        if last_brace_index == -1:
            text = text + " }"

        if last_brace_index != -1:
            text = text[:last_brace_index + 1]

        # Remoce the last comma
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)


        return text

    except Exception as e:
        save_error(str(e), "validate_json", traceback.format_exc())
        return None

def is_valid_json(json_string):
    try:
        json.loads(json_string)
        print("Json is valid")
        return True
    except ValueError:
        print("Json is invalid")
        return False

def chat(message):
    # Use OpenAI Chat API to generate a response
    # Access attributes from default_params instance
    gpt_model = default_params.gpt_model
    temperature = default_params.temperature
    max_tokens = default_params.max_tokens
    top_p = default_params.top_p
    frequency_penalty = default_params.frequency_penalty
    presence_penalty = default_params.presence_penalty

    response = openai.ChatCompletion.create(
        model=gpt_model,  # Use the model name from default_params
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]
    )
    return response['choices'][0]['message']['content'].strip(), response['usage']['completion_tokens'], \
    response['usage']['prompt_tokens']

def gpt_recommend(user_input: str) -> str:
    text = "Feedback: " + user_input + """
            Assign up to """ + str(default_params.number_of_output_skills) + """ skills to the feedback.
            Return response with only list of skills names and two sentance description. The list must be in JSON format examples:
            {Skill 1: Skill Description, Skill N: Skill Description}.
            """

    start = time.time()
    response, completion_tokens, prompt_tokens = chat(text)
    end = time.time()

    #print("GPT first: {} seconds".format(end - start))

    # Try another prompt if the first one doesn't work
    if response == "":
        text = "Choose up to """ + str(default_params.number_of_output_skills) + """ skills to this problem """ + user_input + """
                Return response with only list of skills names (without description) in JSON "Skill number":"skill name" format."""

        response, completion_tokens, prompt_tokens = chat(text)

    json_validation = is_valid_json(response)

    if json_validation is True:
        # Remove prefix from response
        response_json = response[response.find('{'): response.find('}') + 1]

    else:
        response_json = response

    #print(type(response_json))
    #print("GPT formatted json:", response_json)

    return response_json, completion_tokens, prompt_tokens, json_validation
def gpt_feedback(user_input: str) -> str:
    text = "Feedback: " + user_input + """
            Assign up to """ + str(default_params.number_of_output_skills) + """ skills to the feedback.
            Return response with only list of skills names (one sentance). The list must be in JSON format examples:
            {1: Skill 1, N: Skill N}.
            """

    start = time.time()
    response, completion_tokens, prompt_tokens = chat(text)
    end = time.time()

    #print("GPT first: {} seconds".format(end - start))

    # Try another prompt if the first one doesn't work
    if response == "":
        text = "Choose up to """ + str(default_params.number_of_output_skills) + """ skills to this problem """ + user_input + """
                Return response with only list of skills names (without description) in JSON "Skill number":"skill name" format."""

        response, completion_tokens, prompt_tokens = chat(text)

    json_validation = is_valid_json(response)

    if json_validation is True:
        # Remove prefix from response
        response_json = response[response.find('{'): response.find('}') + 1]

    else:
        response_json = response

    #print(type(response_json))
    #print("GPT formatted json:", response_json)

    return response_json, completion_tokens, prompt_tokens, json_validation

def gpt_skills(user_input: str, skills: list, innential_skills: list) -> str:

    print("skills lenght:", len(skills))
    # Check if there are more skills than one

    # Join skills into a string
    skills = ', '.join(skills[:default_params.number_of_output_skills])

    # text = "Describe each skill: " + skills + " with no more than 280 characters and focus only on the skills, how it can help for the problem '" + user_input + "'. Return response in json with skill:description format."  # print"Second GPT input:", text)

    # Prompt for the second GPT
    if len(skills) > 1:
        text_extended = "How  " + str(default_params.number_of_output_skills) +  " skills: " + skills + " ,can help for this problem: " + user_input + ". Return in depth answer in two sentences for each skill in json with skill:description format."
    else:
        text_extended = "How  " + str(default_params.number_of_output_skills) + " skills: " + skills + " ,can help for this problem: " + user_input + ". Return in depth answer in two sentences for each skill in json with skill:description format."

    start = time.time()
    # Call to GPT
    response, completion_tokens, prompt_tokens = chat(text_extended)
    end = time.time()
    print(response)
    print("""GPT second: {} seconds""".format(end - start))

    # Remove prefix from response
    response_json = response[response.find('{'): response.find('}') + 1]

    # print"GPT formatted json:", response_json)
    return response_json, completion_tokens, prompt_tokens

def calc_price(model: str, completion_tokens: int, prompt_tokens: int):
    try:
        price = None

        gpt_4_input_price = 0.03
        gpt_4_output_price = 0.06

        gpt_3_input_price = 0.0015
        gpt_3_output_price = 0.002

        # Calculate price of GPT3 or GPT4
        if model == 'gpt-4':
            price = completion_tokens / 1000 * gpt_4_output_price + prompt_tokens / 1000 * gpt_4_input_price
        elif model == 'gpt-3':
            price = completion_tokens / 1000 * gpt_3_output_price + prompt_tokens / 1000 * gpt_3_input_price

        return price

    except Exception as e:
        save_error(str(e), "calc_price", traceback.format_exc())
        return 0

def process_message(message):
    # Measure total time
    start_total = time.time()
    try:

        # Save input to class
        FeedbackInfo.message = message

        # Get innential skills
        innential_list = innential_skills()

        # Use OpenAI Chat API to generate a response
        first_response, completion_tokens_v1, prompt_tokens, json_validation = gpt_feedback(message)

        if json_validation is True:
            FeedbackInfo.gpt_first_response = json.loads(validate_json(first_response))
        else:
            FeedbackInfo.gpt_first_response = first_response

        # Calculate price of the request
        price_v1 = calc_price('gpt-3', completion_tokens_v1, prompt_tokens)

        # Measure time for assigning skills
        start_skills = time.time()

        # Assign skills
        skill_list, user_vector = return_skills_improved(first_response, json_validation, innential_list)


        FeedbackInfo.skills = skill_list
        FeedbackInfo.user_vector = user_vector

        end_skills = time.time()
        FeedbackInfo.assign_skills_time = end_skills - start_skills

        # GPT call again
        processed_text, completion_tokens_v2, prompt_tokens = gpt_skills(message, [skill for score, skill in skill_list], innential_list)

        #print("Processed text:", processed_text)

        # Remove html tags
        processed_text = remove_html_tags(processed_text)

        #print("Remove html tags:", processed_text)

        processed_text = validate_json(processed_text)

        #print(processed_text)

        # Load response to json
        json_text = json.loads(str(processed_text))

        # Check if the skills from responses matches the skills from innential, if not then remove them
        filtered_skills = {skill: description for skill, description in json_text.items() if
                           skill in innential_list}

        # Save response
        FeedbackInfo.gpt_second_response = filtered_skills

        price_2 = calc_price('gpt-3', completion_tokens_v2, prompt_tokens)

        # Calculate total price
        total_price = round(price_v1 + price_2, 5)
        FeedbackInfo.total_price = total_price
        FeedbackInfo.total_tokens = completion_tokens_v1 + completion_tokens_v2

        response = {
            "Feedback": filtered_skills,
        }

    except Exception as e:
        save_error(str(e), "process_message", traceback.format_exc())
        return

    save_response()

    end_total = time.time()
    print("Total time feedback:", end_total - start_total)

    return response, user_vector

def recommend_feedback(message):
    try:
        # Measure total time
        start_total = time.time()

        # Save input to class
        FeedbackInfo.message = message

        # Get innential skills
        innential_list = innential_skills()

        # Use OpenAI Chat API to generate a response
        first_response, completion_tokens_v1, prompt_tokens, json_validation = gpt_recommend(message)

        if json_validation is True:
            FeedbackInfo.gpt_first_response = json.loads(validate_json(first_response))
        else:
            FeedbackInfo.gpt_first_response = first_response

        # Assign skills
        skill_list, user_vector = return_skills_improved(first_response, json_validation, innential_list)


        FeedbackInfo.skills = skill_list
        FeedbackInfo.user_vector = user_vector

        end = time.time()

        print("total time feedback:", end - start_total)

    except Exception as e:
        save_error(str(e), "process_message", traceback.format_exc())
        return

    return first_response, user_vector


@app.post("/process_message")
def process_message_api(message: Message):
    # Generate user ID
    FeedbackInfo.user_id = generate_user_id()

    response = process_message(message.message)
    return response

@app.post("/feedback_recommendation")
def feedback_recommendation_api(message: Message):
    start = time.time()
    # Generate user ID
    FeedbackInfo.user_id = generate_user_id()

    response, user_vector = recommend_feedback(message.message)

    recommendation = recommendation_engine(user_vector, response, message.message)

    end = time.time()

    print("")
    print("Recommendation time:", end - start)
    print("")

    return response, recommendation

@app.post("/chat_recommendation")
def chat_recommendation(message: Message):
    additional_skill = assign_skills_simple(message.message, innential_skills())
    user_vector = Candidate.user_vector

    print("Additional skill:", additional_skill)

    # Check if additional skill exists and append it to user_vector
    if additional_skill[0][0]:
        if additional_skill[0][0] not in user_vector:
            user_vector.append(additional_skill[0][0])

    top_n_candidates = generate_candidates(user_vector, Candidate.user_feedback, Candidate.user_input, n_candidates=100)
    response = selection(top_n_candidates, message.message, weight=1)
    return response


@app.get('/healthcheck')
def healthcheck():
    return {'status': 'ok'}


# Define an endpoint to access errors
@app.get("/errors")
def get_errors():
    if os.path.exists(ERRORS_FILE):
        with open(ERRORS_FILE, "r") as file:
            errors = json.load(file)
        return errors
    else:
        return []

# Define endpoint to access data from script
@app.get("/response_info")
def get_response():
    if os.path.exists(RESPONSE_FILE):
        with open(RESPONSE_FILE, "r") as file:
            responses = json.load(file)
        return responses
    else:
        return []


# Update API parameters
@app.put("/update_params")
def update_params(params: ApiParams):
    default_params.gpt_model = params.gpt_model
    default_params.temperature = params.temperature
    default_params.max_tokens = params.max_tokens
    default_params.top_p = params.top_p
    default_params.frequency_penalty = params.frequency_penalty
    default_params.presence_penalty = params.presence_penalty
    default_params.bert_model = params.nlp_model
    default_params.z_score_threshold = params.z_score_threshold
    default_params.cutoff_skills = params.cutoff_skills
    default_params.number_of_output_skills = params.number_of_output_skills
    default_params.missing_skills_cutoff = params.missing_skills_cutoff
    default_params.tech_skills_number = params.tech_skills_number
    default_params.soft_skills_number = params.soft_skills_number
    return {"message": "Parameters updated successfully."}


# Reset parameters to default
@app.put("/reset_params")
def reset_params():
    default_params.__init__()
    return {"message": "Parameters reset to default values."}