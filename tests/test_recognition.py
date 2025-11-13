import requests
import os

# Base URL of the running Flask application
BASE_URL = 'http://127.0.0.1:5000'

def login(session, username, password, is_admin=False):
    """Logs in to the application and returns the session."""
    login_url = f"{BASE_URL}/login"
    data = {
        'username': username,
        'password': password
    }
    if is_admin:
        data['is_admin'] = 'true'
    
    response = session.post(login_url, data=data)
    if response.status_code == 200 and 'dashboard' in response.url:
        print(f"Successfully logged in as {username}")
        return session
    else:
        print(f"Failed to log in as {username}")
        return None

def test_recognize(session, file_path):
    """Tests the /recognize endpoint."""
    if not os.path.exists(file_path):
        print(f"Error: Test image not found at {file_path}")
        return

    recognize_url = f"{BASE_URL}/recognize"
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f)}
        response = session.post(recognize_url, files=files)

    print(f"\nTesting /recognize with {os.path.basename(file_path)}...")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response JSON: {response.json()}")
    except requests.exceptions.JSONDecodeError:
        print(f"Response Text: {response.text}")


if __name__ == '__main__':
    # Use a session to persist login
    with requests.Session() as s:
        # Log in as a normal user
        s = login(s, 'user', 'user')

        if s:
            # Path to a test image
            # Using a relative path from the project root
            image_path = os.path.join('data', 'bike.jpg')
            test_recognize(s, image_path)
