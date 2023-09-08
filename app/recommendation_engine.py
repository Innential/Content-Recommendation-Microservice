import json
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import requests
from sentence_transformers import SentenceTransformer, util


# Sentence model tranformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def innential_API():
    # Connect to API and gather all skills to list
    skills = requests.get("https://api.innential.com/scraper/skills-categories/public")

    # Check API call status and raise exception if it was other than 200
    if skills.status_code == 200:
        json_data = skills.json()

    else:
        raise Exception("Failed to retrieve data from API")

    # Iterate over json dict and gather all skills
    subskills = [list(subcategory_dict.keys())[0].lower() for subcategory_list in json_data.values() for subcategory_dict in subcategory_list]

    return subskills

def read_json(path):
    with open(path) as f:
        return json.load(f)

def filter_courses_sbert(feedback, list_of_candidates, model, top_n, cutoff):
    # FIltering based on SBERT

    #print("Course: ", course)
    course = [course for course,_ in list_of_candidates]
    # Encode the text
    embeddings_text = model.encode(feedback)
    embeddings_cat = model.encode(course)

    # Compute cosine similarity between all pairs
    cos_sim = util.cos_sim(embeddings_cat, embeddings_text)

    # Add all pairs to a list with their cosine similarity score
    keys = [[cos_sim[i][0], i, 0] for i in range(len(cos_sim))]

    # Sort list by the highest cosine similarity score
    #keys = sorted(keys, key=lambda x: x[0], reverse=True)

    #print(keys)

    assigned_skills = [[course[i], score] for score, i, j in keys if score > cutoff]

    #print(assigned_skills)

    list_of_candidates = [list(item) for item in list_of_candidates]

    # Add the score from assigned_skills to the score from list_of_candidates
    for i in range(len(list_of_candidates)):
        for j in range(len(assigned_skills)):
            if list_of_candidates[i][0] == assigned_skills[j][0]:
                list_of_candidates[i][1] += assigned_skills[j][1]
                break

    # Sort list_of_candidates by the updated score
    list_of_candidates.sort(key=lambda x: x[1], reverse=True)


    return list_of_candidates[:top_n]


def normalize(data):
    '''
    This function will normalize the input data to be between 0 and 1

    params:
        data (List) : The list of values you want to normalize

    returns:
        The input data normalized between 0 and 1
    '''
    min_val = min(data)
    if min_val < 0:
        data = [x + abs(min_val) for x in data]
    max_val = max(data)
    return [x / max_val for x in data]

user_preferences =  [
      [
        0.75,
        "Product Roadmapping"
      ],
      [
        0.64,
        "Product Development"
      ],
      [
        0.63,
        "Product Management"
      ],
      [
        0.6,
        "Product Lifecycle Management"
      ],
      [
        0.66,
        "Communicating Effectively in a Team"
      ],
      [
        0.48,
        "Teamwork"
      ],
      [
        0.48,
        "Giving Clear Feedback"
      ]
    ]
user_feedback = "My team members told me that I need to become a better product manager and work on roadmapping"

def recommendation_engine(user_preferences, user_feedback):

    print("User vector: ", user_preferences)
    print("User feedback: ", user_feedback)

    innential_skills = innential_API()

    #print("Innential skills:", innential_skills)

    start = time.time()

    #user_vector = np.array([skill[0] if skill[1] in user_preferences else 0 for skill in innential_skills])

    # Create empty vector
    user_vector = np.zeros(len(innential_skills))

    # Max value
    max_weight = max(pref[0] for pref in user_preferences)
    #print("Max weight: ", max_weight)

    # Cutoff
    cutoff = 0

    # Vector for user preferences
    for pref in user_preferences:
        #print("Pref: ", pref[1])
        skill = pref[1].lower()
        if skill in innential_skills:
            # Normalize to 1
            weight = round(pref[0]/ max_weight,2)
            if weight >= cutoff:
                index = innential_skills.index(skill)
                user_vector[index] = weight

    print("User vector: ", user_vector)

    path = "Data/courses_data.json"

    data = read_json(path)

    iteration_limit = 11000
    cosine_similarities = []

    print("")
    print("Candidates")
    for i, course in enumerate(data):
        if i >= iteration_limit:
            break

        # Create a dictionary to store skill weights for the course
        skill_weights = {skill["skill"].lower(): round(skill["score"],2) for skill in course["Skills"]}

        # Vector for the course based on all_skills with corresponding weights
        course_vector = np.array([skill_weights[skill] if skill in skill_weights else 0 for skill in innential_skills])

        # Calculate cosine similarity between course vector and user vector
        cosine_sim = cosine_similarity([course_vector], [user_vector])[0][0]

        # Append course title and cosine similarity to the list
        cosine_similarities.append((course, cosine_sim))

    # Sort the courses based on cosine similarity in descending order
    cosine_similarities.sort(key=lambda x: x[1], reverse=True)

    # Get the top 20 candidates with their respective titles
    top_n_candidates = cosine_similarities[:100]

    # Print the top 20 candidates with their respective titles
    for i, (course, cosine_sim) in enumerate(top_n_candidates):
        print(f"{i+1}. {course['Title']} - Cos: {round(cosine_sim,2)} - {course['Skills']}")

    print("")
    print("Sbert filtering:")

    # Filtering based on SBERT
    start_sbert = time.time()
    filtering = filter_courses_sbert(user_feedback, top_n_candidates, sentence_model, 10, 0.1)
    end_sbert = time.time()

    print(f"Total time taken: {end_sbert - start_sbert} seconds")

    for i, course in enumerate(filtering):
        print(f"{i+1}. {course[0]['Title']} - {course[1]} - {course[0]['Level']} - {course[0]['Skills']}")

    end = time.time()
    print(f"Total time taken: {end - start} seconds")

    return filtering
