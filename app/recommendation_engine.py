import json
import os
import re

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

from config import Innential

# Sentence model tranformer
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")


class Candidate(BaseModel):
    candidates: list
    selection: list
    user_vector: list
    user_input: str
    user_feedback: str


def filter_courses_sbert(feedback, list_of_candidates, model, top_n, cutoff, weight=1.5):
    """
        Filters a list of courses using SBERT embeddings and cosine similarity.
        Args:
            feedback (str): The text feedback to compare against the courses.
            list_of_candidates (List[Tuple[str, float]]): A list of tuples containing the courses and their initial scores.
            model (SBERTModel): The SBERT model used for encoding the text and courses.
            top_n (int): The maximum number of courses to return.
            cutoff (float): The minimum cosine similarity score required for a course to be considered.
            weight (float, optional): The weight to apply to the score from `assigned_skills`. Defaults to 1.5.
        Returns:
            List[Tuple[str, float]]: A list of tuples containing the top `top_n` courses and their scores.
        """

    course = [course for course, _ in list_of_candidates]

    # Encode the text
    embeddings_text = model.encode(feedback)
    embeddings_cat = model.encode(course)

    # Compute cosine similarity between all pairs
    cos_sim = util.cos_sim(embeddings_cat, embeddings_text)

    # Add all pairs to a list with their cosine similarity score
    keys = [[cos_sim[i][0], i, 0] for i in range(len(cos_sim))]

    # Apply cutoff
    assigned_skills = [[course[i], score] for score, i, j in keys if score > cutoff]

    list_of_candidates = [list(item) for item in list_of_candidates]

    # Add the score from assigned_skills to the score from list_of_candidates
    for i in range(len(list_of_candidates)):
        for j in range(len(assigned_skills)):
            if list_of_candidates[i][0] == assigned_skills[j][0]:
                # Add score with the assigned weight
                list_of_candidates[i][1] += assigned_skills[j][1] * weight
                break

    # Sort list_of_candidates by the updated new score
    list_of_candidates.sort(key=lambda x: x[1], reverse=True)

    return list_of_candidates[:top_n]


def read_json_files():
    data_list = []

    file_paths = ['Data/coursera_dry.json', 'Data/datacamp_dry.json', 'Data/pluralsight_prod.json',
                  'Data/udemy_prod.json', 'Data/udemy_prod_2.json']

    absolute_file_paths = [os.path.abspath(path) for path in file_paths]

    # Loop through each file and read its data
    for file_path in absolute_file_paths:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            data_list.extend(data)

    return data_list


def find_skills(user_input, skills):
    """
    Find innential skills using REGEX in the given user input.
    Parameters:
        feedback (str): The user feedback to search for skills.
        skills (list): A list of skills to search for in the feedback.
    Returns:
        list: A list of skills found in the feedback.
    """

    found_skills = []
    for skill in skills:
        # Create a regex pattern for the skill (case-insensitive)
        pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
        if pattern.search(user_input):
            found_skills.append(skill)
    return found_skills


def test_information():
    user_vector = [
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
    user_input = "My team members told me that I need to become a better product manager and work on roadmapping"

    return user_vector, user_input


def generate_candidates(user_preferences, user_feedback, user_input, n_candidates=100):
    """
    Generates candidates based on user preferences, user feedback, and user input.
    Parameters:
    - user_preferences: A list of tuples representing the user's preferences. Each tuple contains a weight and a skill.
    - user_feedback: A string representing the user's feedback.
    - user_input: A string representing the user's input.
    - n_candidates: An optional integer representing the number of candidates to generate. Default is 100.
    Returns:
    - top_n_candidates: A list of tuples representing the top N candidates. Each tuple contains a course and its cosine similarity score.
    """

    # Load innential skills
    innential_skills = Innential.skills

    print(user_preferences)

    Candidate.user_vector = user_preferences

    skills_found = find_skills(user_input, innential_skills)

    print("""Skills found: """, skills_found)

    for skill in skills_found:
        # Check if skill is already in user_preferences
        existing_skills = [pref[1] for pref in user_preferences]
        if skill in existing_skills:
            # Update the weight to 1 for the existing skill
            index = existing_skills.index(skill)
            user_preferences[index] = (1, skill)
        else:
            # Append the new skill with weight 1
            user_preferences.append((1, skill))

    # Sort the user preferences based on the cosine similarity
    user_preferences = sorted(user_preferences, key=lambda x: x[0], reverse=True)

    # Take only the top 8 skills
    user_preferences = user_preferences[:8]

    print("User vector: ", user_preferences)
    print("User feedback: ", user_feedback)

    # Create empty vector
    user_vector = np.zeros(len(innential_skills))

    # Max value for normalization
    max_weight = max(pref[0] for pref in user_preferences)

    # Cutoff
    cutoff = 0

    # Vector for user preferences
    for pref in user_preferences:
        skill = pref[1]
        if skill in innential_skills:
            # Normalize to 1
            weight = round(pref[0] / max_weight, 2)
            if weight >= cutoff:
                index = innential_skills.index(skill)
                user_vector[index] = weight

    print("User vector: ", user_vector)

    # Read courses data from json
    data = read_json_files()

    # Remove duplicates from the initial list of courses
    unique_data = []
    seen_titles = set()
    for course in data:
        title = course["course_title"]
        if title not in seen_titles:
            seen_titles.add(title)
            unique_data.append(course)

    cosine_similarities = []

    print("")
    print("Candidates")
    for course in unique_data:
        # Create a dictionary to store skill weights for the course
        skill_weights = course["analysis_results"]

        # Vector for the course based on all_skills with corresponding weights
        course_vector = np.array([skill_weights[skill] if skill in skill_weights else 0 for skill in innential_skills])

        # Calculate cosine similarity between course vector and user vector
        cosine_sim = cosine_similarity([course_vector], [user_vector])[0][0]

        # Append course title and cosine similarity to the list
        cosine_similarities.append((course, cosine_sim))

    # Sort the courses based on cosine similarity in descending order
    cosine_similarities.sort(key=lambda x: x[1], reverse=True)

    # Get the top N candidates with their respective titles
    top_n_candidates = cosine_similarities[:n_candidates]

    # Save the top N candidates to Innential Class
    Candidate.candidates = top_n_candidates

    return top_n_candidates


def selection(top_n_candidates, user_input, weight, n_candidates=10):
    """
        From the previously generated candidates, select the top 10 based on the SBERT filtering using feedback from GPT

        Parameters:
        - top_n_candidates (list): A list of candidates to be printed and filtered.
        - user_input (str): The user input used for filtering.
        - weight (float): The weight used for filtering.

        Returns:
        - filtering (list): The filtered list of candidates.
        """

    # Print the top N candidates with their respective titles
    for i, (course, cosine_sim) in enumerate(top_n_candidates):
        print(f"{i + 1}. {course['course_title']} - Cos: {round(cosine_sim, 2)} - {course['analysis_results']}")

    print("")
    print("Sbert filtering:")

    # Filtering based on SBERT
    filtering = filter_courses_sbert(user_input, top_n_candidates, sentence_model, n_candidates, 0.1, weight)
    Candidate.selection = filtering  # Save the filtered courses

    for i, course in enumerate(filtering):
        print(
            f"{i + 1}. {course[0]['course_title']} - {course[1]} - {course[0]['analysis_results']} - {course[0]['source_url']}")

    return filtering


def recommendation_engine(user_preferences, user_feedback, user_input):
    """
       Generates recommendations based on user preferences, feedback, and input.
       Parameters:
           user_preferences (list): A list of user preferences.
           user_feedback (dict): A dictionary containing user feedback.
           user_input (str): The user input.
       Returns:
           tuple: A tuple containing the filtered recommendations and the top N candidates.
       """
    # Store user_input and user feedback for narrowing feature
    Candidate.user_input = user_input
    Candidate.user_feedback = user_feedback

    # Generate candidates
    top_n_candidates = generate_candidates(user_preferences, user_feedback, user_input, n_candidates=100)
    # Select the top N
    filtering = selection(top_n_candidates, user_input, weight=1.5, n_candidates=10)

    return filtering, top_n_candidates
