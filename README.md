# RAG Training
A repository designed to practice and train Retrieval-Augmented Generation (RAG) models effectively.

## Overview
This repository is structured to provide the necessary resources and organization for working with RAG models. It includes directories for data storage, preprocessing, model interaction, and deploying the required infrastructure.

### Purpose
The main purpose of this repository is:
- To provide a streamlined workflow for training RAG models.
- To facilitate the creation of embeddings, preprocessing of data, and interaction with the trained models.
- To define and deploy the infrastructure required for efficient use of RAG models, such as vector databases and containers.

---

## Directory Structure
Here’s an overview of the key directories within the repository:

### `Data`
- **Purpose**: Serves as the central storage location for all the documents and data required for training.
- **Content**: This directory includes raw text data, documents, and any other inputs to be used during the training phase.

### `Preprocessing`
- **Purpose**: Houses scripts and utilities for generating embeddings and preparing data for model training.
- **Content**: Preprocessing tools that transform raw data into embeddings optimized for use with RAG models.

### `RAG`
- **Purpose**: Facilitates client-server interaction with the trained RAG model, ensuring smooth and efficient communication.
- **Content**: Scripts and files enabling you to interact with the models, such as APIs, helper scripts, or example usage clients.

### `Infrastructure`
- **Purpose**: Contains definitions and configurations for essential infrastructure components.
- **Content**:
    - Templates or scripts for setting up vector databases (Vector DB).
    - Containerization solutions (e.g., Docker configurations) for deploying and maintaining the RAG-related infrastructure.

---

## How To Use This Repository
1. **Add Your Data**: Place the documents and datasets in the `Data` directory.
2. **Preprocess**: Run scripts from the `Preprocessing` directory to generate embeddings or prepare your data for training.
3. **Train and Interact**: Use the files from the `RAG` directory to train the RAG model or test client-server interactions.