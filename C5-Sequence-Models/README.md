## Course 5: Sequence Models
The fifth course in the Deep Learning Specialization focuses on sequence models and their exciting applications such as speech recognition, music synthesis, chatbots, machine translation, natural language processing (NLP), and more.

### Week 1: Recurrent Neural Networks
Discover recurrent neural networks (RNNs) and several of their variants, including LSTMs, GRUs and Bidirectional RNNs, all models that perform exceptionally well on temporal data.

- Define notation for building sequence models
- Describe the architecture of a basic RNN
- Identify the main components of an LSTM
- Implement backpropagation through time for a basic RNN and an LSTM
- Give examples of several types of RNN
- Build a character-level text generation model using an RNN
- Store text data for processing using an RNN
- Sample novel sequences in an RNN
- Explain the vanishing/exploding gradient problem in RNNs
- Apply gradient clipping as a solution for exploding gradients
- Describe the architecture of a GRU
- Use a bidirectional RNN to take information from two points of a sequence
- Stack multiple RNNs on top of each other to create a deep RNN
- Use the flexible Functional API to create complex models
- Generate jazz music with deep learning
- Apply an LSTM to a music generation task

[Lecture Notes][L1]  
[Assignment: Building a Recurrent Neural Network - Step by Step][C5W1A1]  
[Assignment: Character-level language model][C5W1A2]  
[Assignment: Jazz improvisation with LSTM][C5W1A3]   

### Week 2: Natural Language Processing and Word Embeddings
Use word vector representations and embedding layers to train recurrent neural networks with an outstanding performance across a wide variety of applications, including sentiment analysis, named entity recognition, and neural machine translation.

- Explain how word embeddings capture relationships between words
- Load pre-trained word vectors
- Measure similarity between word vectors using cosine similarity
- Use word embeddings to solve word analogy problems such as Man is to Woman as King is to ______.
- Reduce bias in word embeddings
- Create an embedding layer in Keras with pre-trained word vectors
- Describe how negative sampling learns word vectors more efficiently than other methods
- Explain the advantages and disadvantages of the GloVe algorithm
- Build a sentiment classifier using word embeddings
- Build and train a more sophisticated classifier using an LSTM

[Lecture Notes][L2]  
[Assignment: Word Vector Representation and Debiasing][C5W2A1]  
[Assignment: Emojify!][C5W2A2]  

### Week 3: Sequence Models and the Attention Mechanism
Improve sequence models with the attention mechanism, an algorithm that helps a model decide where to focus its attention given a sequence of inputs; explore speech recognition and how to deal with audio data.

- Describe a basic sequence-to-sequence model
- Compare and contrast several different algorithms for language translation
- Optimize beam search and analyze it for errors
- Use beam search to identify likely translations
- Apply BLEU score to machine-translated text
- Implement an attention model
- Train a trigger word detection model and make predictions
- Synthesize and process audio recordings to create train/dev datasets
- Structure a speech recognition project

[Lecture Notes][L3]  
[Assignment: Neural Machine Translation with Attention][C5W3A1]  
[Assignment: Trigger Word Detection][C5W3A2]  

### Week 4: Transformers
Build the transformer architecture and tackle natural language processing (NLP) tasks such as attention models, named entity recognition (NER) and Question Answering (QA).

- Create positional encodings to capture sequential relationships in data
- Calculate scaled dot-product self-attention with word embeddings
- Implement masked multi-head attention
- Build and train a Transformer model
- Fine-tune a pre-trained transformer model for Named Entity Recognition
- Fine-tune a pre-trained transformer model for Question Answering
- Implement a QA model in TensorFlow and PyTorch
- Fine-tune a pre-trained transformer model to a custom dataset
- Perform extractive Question Answering

[Lecture Notes][L4]  
[Assignment: Transformer Network][C5W4A1]  
[Lab: Transformer Network Preprocessing][C5W4U1]  
[Lab: Transformer Network Application: Named-Entity Recognition][C5W4U2]  
[Lab: Transformer Network Application: Question Answering][C5W4U3]  

### Reference
[Coursera - Sequence Models](https://www.coursera.org/learn/nlp-sequence-models?specialization=deep-learning)


[L1]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W1-Recurrent-Neural-Networks/C5_W1.pdf
[L2]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W2-Introduction-to-Word-Embeddings/C5_W2.pdf
[L3]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W3-Sequence-Models-Attention-Mechanism/C5_W3.pdf
[L4]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W4-Transformer-Network/C5_W4.pdf

[C5W1A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W1-Recurrent-Neural-Networks/A1/Building_a_Recurrent_Neural_Network_Step_by_Step.ipynb
[C5W1A2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W1-Recurrent-Neural-Networks/A2/Dinosaurus_Island_Character_level_language_model.ipynb
[C5W1A3]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W1-Recurrent-Neural-Networks/A3/Improvise_a_Jazz_Solo_with_an_LSTM_Network_v4.ipynb
[C5W2A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W2-Introduction-to-Word-Embeddings/A1/Operations_on_word_vectors_v2a.ipynb
[C5W2A2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W2-Introduction-to-Word-Embeddings/A2/Emoji_v3a.ipynb
[C5W3A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W3-Sequence-Models-Attention-Mechanism/A1/Neural_machine_translation_with_attention_v4a.ipynb
[C5W3A2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W3-Sequence-Models-Attention-Mechanism/A2/Trigger_word_detection_v2a.ipynb
[C5W4A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W4-Transformer-Network/A1/C5_W4_A1_Transformer_Subclass_v1.ipynb
[C5W4U1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W4-Transformer-Network/U1/Embedding_plus_Positional_encoding.ipynb
[C5W4U2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W4-Transformer-Network/U2/Transformer_application_Named_Entity_Recognition.ipynb
[C5W4U3]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C5-Sequence-Models/W4-Transformer-Network/U3/QA_dataset.ipynb
