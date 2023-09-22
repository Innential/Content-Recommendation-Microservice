# REM - Recommendation Engine Microservice
REM is the NLP-based microsrvice that will recommend a content based on the description from the users input. The application uses FeedbackGPT, SBERT and Cosine similiarites to fit the best possible courses for the user.

### Design overview
<img src="https://github.com/Innential/Content-Recommendation-Microservice/blob/main/app/Data/design.jpg" width="800"/>

FeedbackGPT powered by ChatGPT, generates the feedback for the users input. The feedback consist of a few sentences describing the solution and skills for the users question. Example:
```
User feedback:  {
  "Skill 1": "Python Programming: Learn the basics of Python programming language, which is widely used in machine learning.",
  "Skill 2": "Mathematics and Statistics: Gain a solid foundation in mathematics and statistics, as they form the basis of many machine learning algorithms.",
  "Skill 3": "Data Analysis and Visualization: Learn how to analyze and visualize data using tools like pandas, numpy, and matplotlib, which are commonly used in machine learning projects."
}
```
Then using SBERT it assigns skills from Innential Database to sentences above creating `User vector`. Each sentence has it's own similarity score which varies from 0 to 1. Example:
```
 [(1, 'Machine Learning'), [0.68, 'Python'], [0.63, 'Data Visualization'], [0.58, 'Mathematics'], [0.54, 'Machine Learning Algorithms'], [0.52, 'Data Analysis'], [0.5, 'Statistics'], [0.46, 'Machine Learning Frameworks']]
``` 
The courses from Innential database also have assigned skills in the same manner, so it can create a `Course vector`.

### Recommendation engine microservice
REM consist of three main stages:
1. Candidate generation - The algorithm calculates the cosine similarity between users vector and each course vector. The highest the score the course is more similar to the users input.
2. Scoring - The algorithm will calculate cosine similarity between candidates and gpt feedback
3. Re-Ranking - Based on the equation `Candidates Score x Weight + Scoring * Weight` the list will be re-ranked

# Usage
Application provides two Endpoints:
- `/feedback_recommendation` - This API endpoint is designed to provide course recommendations based on user feedback and preferences. It generates recommendations and returns them along with additional information.
- `/chat_recommendation` - This API is designed to have a chat with user. Once the general question was asked to `/feedback_recommendation`, we can narrow down the results.


### Docker deployment
1. Install docker 
2. Create `.env` file in `app` project directory
3. Add the following variable inside `.env`: `OPENAI_API_KEY=<your-openai-api-key>` and replace your Open AI key
4. Run `docker build -t myimage .`
5. Run `docker run -d --name mycontainer -p 80:80 myimage`
7. Go to http://127.0.0.1/docs for interactive API to see the results

### Virtual Env installation
1. Checkout to `main` branch
2. Clone repository
3. Create virtual env ```python -m venv venv```
4. Activete virtual env ```source venv/bin/activate```
5. Install the dependencies ```pip install -r requirements.txt```

### Running FastAPI
1. Create `.env` file in project directory
2. Add the following variable inside `.env`: `OPENAI_API_KEY=<your-openai-api-key>` and replace your Open AI key
3. Start FastAPI server `uvicorn main:app --reload`

### Deployment documentation
- Deployments concepts https://fastapi.tiangolo.com/deployment/concepts/
- Docker deployment documentation https://fastapi.tiangolo.com/deployment/docker/

### API interactive documentation
- Interactive API documentation: http://localhost:8000/docs
- ReDoc API documentation: http://localhost:8000/redoc
- OpenAPI JSON schema: http://localhost:8000/openapi.json


