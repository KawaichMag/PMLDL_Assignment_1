FROM ghcr.io/astral-sh/uv:python3.13-alpine

ADD . /app

WORKDIR /app
RUN uv sync --locked

EXPOSE 8000
CMD ["uv", "run", "fastapi", "run"]