## Introduction

Dude Where's My Lambo answers the question: If I had bought a crypto currency when it first appeared on the public exchange, would I have enough money to buy a lambo if I sold my coins in the last month?

This is the backend for the system

## Requirements
* Python3
* Pipenv

## Getting started

1. Source the virtual environment ```[pipenv shell]```
2. Install the dependencies ```[pipenv install]```


## Run the application
You will need two terminals pointed to the frontend and backend directories to start the servers for this application.

1. To run the backend, make sure you have started the pipenv shell using ```[pipenv shell]```
2. Run this command to start the backend server: ```[python run.py]``` (You have to run this command while you are sourced into the virtual environment)


## Test the application
All the tests for this api are functional tests using pytest

1. To run the test suite, use the following command:  ```[python -m pytest]```


## Bringing the code to production
This code encompasses the backend for a larger system, in the diagram below I have set out the architecture that could be used to create an enterprise ready system that meets all the needs of a high availability and low latency web app.

I have used GCP for the example, but have chosen the components to be as simplistic as possible, to most easily be portable to other cloud providers using some kind of resource management language like terraform

This architecture would use github actions as its CI/CD pipeline using the included push.yml file, but could be updated to use something else within the CI/CD ecosystem like Jenkins

![Alt text](DudeWheresMyLambo.Architecture.png?raw=true "Title")

