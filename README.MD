# Neural Network Experiments

A curated collection of experiments across classical deep learning, PyTorch/TensorFlow fundamentals, computer vision, time series, autoencoders, self-organizing maps, and Generative AI (LangChain + LLMs). The goal is breadth over depth, showcasing techniques, tooling, and practical workflows.

## Highlights: Skills and Technologies

- Programming: Python
- Deep Learning: PyTorch, Lightning (training loops, checkpoints), TensorFlow/Keras
- ML: scikit-learn (preprocessing, metrics, model selection), imbalanced data handling (oversampling)
- Data: NumPy, pandas
- Visualization: Matplotlib, Seaborn
- Generative AI: LangChain (chains, memory, RAG), LLMs (OpenAI, Ollama), Chroma vector DB, Hugging Face (transformers, diffusers)
- MLOps-adjacent: experiment logging/checkpointing, dataset loaders, prompt engineering patterns

## Project Index (Overview)

Fundamentals
- Tensors and PyTorch basics: [Tensors.ipynb](Tensors.ipynb)
- Simple PyTorch model/regression: [pytorch/model.ipynb](pytorch/model.ipynb)

Supervised Learning (Tabular)
- ANN (Keras) for churn-style prediction: [artificial_neural_network.ipynb](artificial_neural_network.ipynb)
- Feature engineering + model evaluation (ROC/PR/CM) on churn: [churn.ipynb](churn.ipynb)
- Diabetes classification (scaling, splits, oversampling): [diabetes_neural_networks.ipynb](diabetes_neural_networks.ipynb)
- Obesity prediction with ANN: [Obesity_ANN.ipynb](Obesity_ANN.ipynb)
- Wine reviews/text experiments: [nn_wine_review.ipynb](nn_wine_review.ipynb)

Computer Vision
- CNN (Keras) with data generators (cats vs dogs): [cnn.ipynb](cnn.ipynb)
- MNIST CNN (preprocessing, training/eval): [MNIST_cnn.ipynb](MNIST_cnn.ipynb)

Time Series
- LSTM for ETH price predictions: [LSTM-ETH_predictions.ipynb](LSTM-ETH_predictions.ipynb)

Representation Learning and Unsupervised
- Stacked Autoencoder (PyTorch): [Autoenconders.ipynb](Autoenconders.ipynb)
- Self-Organizing Maps on fashion data: [SOM_fashion.ipynb](SOM_fashion.ipynb)

PyTorch Lightning and Custom Architectures
- Multi-input classification (Iris) with Lightning modules, dataloaders, checkpoints: [nn/multi_inputs.ipynb](nn/multi_inputs.ipynb)

Diffusion and Hugging Face
- Diffusers quickstart: [nn/diffusers.ipynb](nn/diffusers.ipynb)
- Hugging Face Diffusers pipelines: [nn/hf_diffusers.ipynb](nn/hf_diffusers.ipynb)
- HF pipelines/experiments: [nn/hf_pipes.ipynb](nn/hf_pipes.ipynb)
- Utility helpers: [nn/func.py](nn/func.py)

Generative AI, RAG, and Prompting
- RAG pipeline experiments (retriever, embeddings, vector store, LLM): [rag_pipeline.ipynb](rag_pipeline.ipynb)
- Conversational memory variants (buffer window, history, summaries): [langchain/memory_langchain.ipynb](langchain/memory_langchain.ipynb)
- Prompting patterns and assistant/system scaffolds: [langchain/prompt.ipynb](langchain/prompt.ipynb)
- RAG app scaffolding and tests: [rag_app/](rag_app/) • Example utilities/chain wiring: [rag_chain.py](rag_chain.py)

## What This Repository Demonstrates

- Data handling: scaling, encoding, train/validation/test splits, class balancing
- Model building: ANN/CNN/LSTM/AE architectures in Keras and PyTorch
- Training workflows: Lightning modules, custom training steps, checkpoints, and evaluation loops
- Evaluation: accuracy, ROC/AUC, precision/recall, confusion matrices, visual diagnostics
- Generative AI: end-to-end RAG (document loaders, splitters, embeddings, retriever-LLM chains), conversational memory, prompt templates
- Ecosystem know-how: integrating Hugging Face, Chroma DB, OpenAI/Ollama backends

## Running the Projects

- Environment
  - Python 3.10+ recommended
  - Install packages used across notebooks:
    - Core: torch, torchvision, lightning, tensorflow, keras, scikit-learn, numpy, pandas, matplotlib, seaborn
    - GenAI: langchain, langchain-openai, langchain-ollama, langchain-text-splitters, chromadb, huggingface_hub, transformers, diffusers, python-dotenv
- Notebooks
  - Open any notebook (e.g., [cnn.ipynb](cnn.ipynb), [MNIST_cnn.ipynb](MNIST_cnn.ipynb), [nn/multi_inputs.ipynb](nn/multi_inputs.ipynb)) in VS Code and run cells top-to-bottom.
- RAG/LLM experiments
  - Set environment variables in [.env](langchain/.env) (API keys, model hosts)
  - Start Ollama (if using local LLMs) or configure OpenAI
  - Explore [rag_pipeline.ipynb](rag_pipeline.ipynb), [langchain/memory_langchain.ipynb](langchain/memory_langchain.ipynb), and [langchain/prompt.ipynb](langchain/prompt.ipynb)

## Repo Notes

- Editor settings: [.vscode/settings.json](.vscode/settings.json)
- Ignore rules: [.gitignore](.gitignore), [nn/.gitignore](nn/.gitignore), [langchain/.gitignore](langchain/.gitignore)
