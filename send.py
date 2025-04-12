import requests
import json

# Define the URL and payload
url = "https://advana-challenge-check-api-cr-k4hdbggvoq-uc.a.run.app/software-engineer"
payload = {
    "name": "Leonardo Pacheco",
    "mail": "wleonardop@gmail.com",
    "github_url": "https://github.com/wleopach/ChallengeLatam.git",
    "api_url": "https://demo-1-684881852527.us-central1.run.app"
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the status code and response text for debugging
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

# Check if the response matches the expected one
expected_response = {
    "status": "OK",
    "detail": "your request was received"
}

# Convert the response text into a dictionary
response_json = response.json()

# Check if the response is as expected
if response_json == expected_response:
    print("Success: The response matches the expected result.")
else:
    print("Error: The response does not match the expected result.")
    print(f"Expected: {expected_response}")
    print(f"Received: {response_json}")
