FROM python:3.10-slim
ENV PYTHONUNBUFFERED True
ENV APP_HOME /
WORKDIR $APP_HOME
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "run.py" ]
