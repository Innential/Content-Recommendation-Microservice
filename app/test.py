import requests
import json
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

# Check if the request was successful (HTTP status code 200)
if response.status_code == 200:
    # The API response content (JSON data) is stored in response.json()
    data = response.json()
    # You can now work with the data as needed
    print(data[10])
else:
    # If the request was not successful, print an error message
    print(f"Error: {response.status_code}")
    print(response.text)

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

save_to_json(data, "data.json")