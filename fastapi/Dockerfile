# docker build -t simpson-classifier .
# docker run -p 5000:80 simpson-classifier

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Add requirements.txt before rest of repo for caching
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt

COPY ./app /app