import requests

url = "https://www.brookings.edu/topic/artificial-intelligence/feed/"
try:
    r = requests.get(url, timeout=20)
    print("status", r.status_code)
    text = r.text
    lines = text.splitlines()
    for i, l in enumerate(lines[:200], start=1):
        print(f"{i:03}: {l}")
except Exception as e:
    print('fetch error', e)
