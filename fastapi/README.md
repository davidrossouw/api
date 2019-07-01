# Instructions
From folder `api/fastapi/app` run `uvicorn main:app --reload` to start the server.

To classify a Simpson character, POST your image as follows:

```
path = 'nelson_muntz_40.jpg'

with open(path, "rb") as f:
    img_bytes = f.read()

file = {"file": img_bytes}

# post request
response = requests.post(url="http://localhost:8000/classify_image/", files=file)

# decode response
print(json.loads(response.text))
```