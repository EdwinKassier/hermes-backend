# Use an official Python runtime as a parent image
FROM python:3.10-slim

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8080

# Define the command to run your Flask application
CMD ["python", "run.py"]
