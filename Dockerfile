FROM python:3.9
WORKDIR /app
COPY streamlit_app.py /app/streamlit_app.py
COPY requirements_chatbot.txt /app/requirements_chatbot.txt
RUN pip install -r requirements_chatbot.txt
EXPOSE 8080
CMD ["streamlit", "run", "/app/streamlit_app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]

