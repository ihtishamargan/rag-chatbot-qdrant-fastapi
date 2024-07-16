## RAG backend boilerplate project

This project uses Qdrant (vector store), langchain, openai (embedding & generation models) and FastApi to build a basic RAG pipeline.

## Features

- **Google Drive Integration**: Load documents from Google Drive folders.
- **Text Splitting**: Splits documents into smaller chunks for efficient processing.
- **Qdrant Vector Store**: Utilizes Qdrant for storing and retrieving text vectors.
- **FastAPI Backend**: Provides RESTful endpoints for ingestion and retrieval of documents.
## Installation

### Prerequisites

The project uses [PDM](https://github.com/pdm-project/pdm) for package management, so make sure this is also available on your system.

```shell
brew install pdm
```

### Setup

1. Clone the repository
2. Inside the repository, execute the PDM install command:
   ```shell
      pdm install
   ```
   This command will create a virtual environment under the `.venv` folder, which is git ignored, and install the dependencies inside it.



### Configuration

The application can be configured using environment variables. See `src/config.py` for more details.

## Commands

### Start

To execute the application execute:

```shell
pdm run start
```

This will run the PDM script called `start`, which you can see defined in the [pyproject.toml file](./pyproject.toml)

### Lint

To execute all the linters and formatters run:

```shell
pdm run lint
```

Which will run [pre-commit](https://pre-commit.com)

## Docker

The application uses docker. To run it locally using docker compose, run:

```shell
docker compose up
```

You can always rebuild the docker image with:
```shell
docker compose build
```

Or build and run it with:
```shell
docker compose up --build
```
