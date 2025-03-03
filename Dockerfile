FROM continuumio/miniconda3

WORKDIR /home
ENV PYTHONPATH=/home

# Update package list
RUN apt-get update

# Install bash and other required tools
RUN apt-get install -y bash nano unzip curl

# Install the Deta CLI
RUN curl -fsSL https://get.deta.dev/cli.sh | bash

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python", "app/train.py"]
