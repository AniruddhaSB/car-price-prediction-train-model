#Use an official Python runtime as a parent image.
#We're using a slim version to keep the image size small.
FROM python:3.9-slim-bullseye

#Set the working directory inside the container.
WORKDIR /app

#Copy the dependencies file into the container at /app.
COPY requirements.txt .

#Install any dependencies specified in requirements.txt.
#We use --no-cache-dir to keep the image size down.
RUN pip install --no-cache-dir -r requirements.txt

#Copy the rest of the application's source code to the container.
COPY . .

#Expose port 8080.
#This tells Docker that the container listens on this port.
EXPOSE 8080

#Define the command to run the application.
#This is the command that gets executed when the container starts.
#For a simple app, this is sufficient. For production, consider using a production-grade
#server like Gunicorn (e.g., CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]).
CMD ["python", "app.py"]