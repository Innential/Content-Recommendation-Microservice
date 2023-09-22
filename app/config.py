import requests
from fastapi import HTTPException
from fastapi_utils.tasks import repeat_every
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


######################### CLASSES ############################

# Default parameters
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


# Innential skills class
class Innential(BaseModel):
    skills: list
    courses: list

# TODO
# Another func for endpoint for innential

######################### FUNCTIONS ############################
@repeat_every(seconds=24 * 60 * 60)  # 24 hours
async def innential_skills():
    """
    An async function that retrieves the skills categories from the Innential API and saves them to the Innential.skills attribute.
    HTTPException: If an error occurs while making the API request.
    """
    try:
        skills = requests.get("https://api.innential.com/scraper/skills-categories/public")
        skills.raise_for_status()
        json_data = skills.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail="An error occurred and has been saved.")

    subcategories = [list(subcategory_dict.keys())[0] for subcategory_list in json_data.values() for subcategory_dict in
                     subcategory_list]

    Innential.skills = subcategories

@repeat_every(seconds=24 * 60 * 60)  # 24 hours
async def innential_courses():
    # API endpoint URL
    url = "https://api.innential.com/scraper/items-list"

    # Define the parameters
    params = {
        "filter": "datacamp",  # Change this to your desired filter values
        "type": "e-learning",       # Change this to your desired type value
        "complete": 1               # Change this to your desired complete value
    }

    # Send a GET request to the API
    response = requests.get(url, params=params)

    if response.status_code == 200:
        # The API response content (JSON data) is stored in response.json()
        data = response.json()
        Innential.courses = data
        # You can now work with the data as needed
        print("Course from Innential Database:", data[10])
    else:
        # If the request was not successful, print an error message
        print(f"Error: {response.status_code}")
        print(response.text)





######################### VARIABLES ############################
ERRORS_FILE = "errors.json"
RESPONSE_FILE = "response.json"

# Default parameters instance
default_params = DefaultParams()

# Set up SentenceTransformer model
nlp_model = SentenceTransformer(default_params.bert_model)
