FROM python:3.10-slim-buster
RUN pip install uv

ADD . /app

WORKDIR /app
RUN uv sync

EXPOSE 8000
CMD ["uv", "run", "fastapi", "run"]