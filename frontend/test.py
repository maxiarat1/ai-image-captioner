import requests
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/generate',
        files={'image': f},
        data={'model': 'wdvit', 'prompt': 'Describe this image'}
    )
print(response.json()['caption'])