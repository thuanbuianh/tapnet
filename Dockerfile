FROM python:latest
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8051
CMD ["streamlit", "run", "inference.py", "--server.port=8501", "--server.address=0.0.0.0"]
