# Use an official Python runtime as a base image
FROM python:3.10.14

# Set the working directory
WORKDIR /RAG-Implementation

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python main.py

# Copy the bot's source code
COPY . .

# Set the command to run the bot
CMD ["python", "bot.py"]
