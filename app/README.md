## Files
- `main.py` - FASTAPI app with endpoints and FeedbackGPT microservice
- `recommandation_engine.py` - algorithms that calculates content recommendation for user
- `config.py` - define classes and paramaters, also have two async functions for gathering Innentials skills and content from database

## `Recommendation_engine.py`
There two important functions:
- `generate_candidates` - generates a list of candidates only based on their cosine score in relation to user vector
- `selection`- generates a narrow list of top 10 candidates by calculating their cosine score between course description and user input or user feedback

### Function `generate_candidates`
1. Input Parameters:
- user_preferences: A list of tuples representing the user's preferences.
- user_feedback: A string representing the user's feedback.
- user_input: A string representing the user's input.
- n_candidates (optional, default=100): An integer representing the number of candidates to generate.
2. Load skills categories from the `Innential.skills` attribute.
3. If any skills mentioned in the user_input are already in the user_preferences. If a skill is found, its weight is updated to 1. If not, the skill is added to user_preferences with a weight of 1.
4. The top 8 skills from the sorted `user_preferences` list are retained, and the rest are discarded.
5. An empty vector `user_vector` is created with the length of Innential skills.
6. Weights in `user_preferences` are normalized to a maximum value of 1, and values below a cutoff threshold are set to 0.
7. Load courses
8. For each unique course, a cosine similarity score is computed between the course's skill vector and the user's skill vector.
9. The top n_candidates courses with the highest cosine similarity scores are selected as top candidates.

### Function `selection`
1 Input Parameters:
- top_n_candidates (list): A list of candidates to be filtered.
- user_input (str): The user input used for filtering.
- weight (float): The weight used for filtering.
- n_candidates (optional, default=10): An integer representing the maximum number of candidates to select.
2. S-BERT Filtering: The function performs filtering on the top candidates based on SBERT embeddings and user input using the filter_courses_sbert function. It considers the provided user_input, top_n_candidates, and the specified weight.
3. The filtered courses are saved to the `Candidate.selection` attribute, making them available for further use.


