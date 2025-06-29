FROM python:3.12-slim-bullseye

WORKDIR /code

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl vim \
    && apt-get install -y --reinstall procps

RUN --mount=type=secret,id=env,target=/code/.env \
    export $(cat /code/.env | xargs) \
    && git config --global user.name $GIT_USERNAME \
    && git config --global user.email $GIT_EMAIL

COPY ./requirements.txt /code
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /code

EXPOSE 8000

CMD ["sleep", "infinity"]
