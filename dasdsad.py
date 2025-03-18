import requests
urls = ["https://example.com"]
for url in urls:
    response = requests.get(url)
    print(f"{url} - Status Code: {response.status_code}")