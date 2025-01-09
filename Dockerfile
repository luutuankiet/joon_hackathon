FROM python:3.9
WORKDIR /app
COPY chatbot.py index.html requirements_chatbot.txt .
RUN pip install gunicorn
RUN pip install -r requirements_chatbot.txt
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app
