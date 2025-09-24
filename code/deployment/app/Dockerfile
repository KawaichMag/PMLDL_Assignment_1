FROM ghcr.io/astral-sh/uv:python3.13-alpine

ADD . /app

WORKDIR /app
RUN uv sync --locked
RUN uv tool install streamlit

EXPOSE 8501
CMD ["streamlit", "run", "main.py"]