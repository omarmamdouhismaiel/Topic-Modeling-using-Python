# Topic-Modeling-using-Python
Using Topic Models in Latent-Variables Cluster-Analysis, Information Retrieval (search), Recommender System applications.

Topic models with SVD matrix factorization which used in 'Latent Semantic Analysis (LSA)' Algorithm are popular in Information Retrieval.

Topic models with a generative probabilistic model, a bagofwords model, Latent Dirichlet Allocation (LDA) Algorithm are popular in Latent Variables/Topics and structure Discovering (Semantic Analysis) or Topics/Words distributions Discovering. And in soft/fuzzy clustering whether in word clusters represents the topics as multinomial distribution of words, or in topic cluster represents the document as multinomial distribution of topics.

LDA = probabilistic version of LSA = a fully Bayesian version of pLSI
LSA = PCA applied to text data = linear algebra method = LSI Latent Semantic Indexing

Topic models with SVD matrix factorization which used in 'Latent Semantic Analysis (LSA)' Algorithm are popular in Information Retrieval.

Topic models with a generative probabilistic model, a bagofwords model, Latent Dirichlet Allocation (LDA) Algorithm are popular in Latent Topics Discovering (Semantic Analysis).or Topics/Words distributions Discovering.

LDA = probabilistic version of LSA = a fully Bayesian version of pLSI
LSA = PCA applied to text data = linear algebra method = LSI Latent Semantic Indexing

LDA Algorithm (Iterative Algorithm)
1. Initialize parameters.
2. Initialize topic assignments randomly.
3. Iterate: 
                For each word in each document:
                       Resample topic for word, giving all other words and their current topic assignments.
4. Get results.
5. Evaluate model.

Topic Models are documents soft-clusters representing them as multinomial distribution of topics. Every topic represents as multinomial distribution of words.
We can see them also as Dimensionality Reduction models, which transform our documents from word-vector to topic-vector. 


LDA topic model uses posterior/probabilistic inference to figure out what are the missing pieces of story (latent variables) that best explain our data, through:
- Topics to word types _ multinomial distribution.
- Documents to topics _ multinomial distribution.
So, 
Model = data story 
Topics = latent variables = missing pieces of story 
Statistical Inference = filling in those missing pieces

Generative Topic Models are language models, so it can be used in NLP applications.

Each document gets represented as a pattern of LDA topics (multinomial distribution). Making Every document appear different enough to be separable, similar enough to be grouped, like fingerprint or DNA.

We can use Jensen-Shannon Distance 'Similarity Metric' = root of (Jesnen-Shannon Divergence) between documents represented as multinomial/probability distribution, gives value between 0 and 1. Near 0 means similar, near 1 means different. We can get similar documents from your corpus based on a threshold gets by experimenting in recommendations.

Hyperparameter = k, alpha, beta, (similarity distance threshold possible)
Parameters = theta, phai

Use perplexity to see if your model is representative of the documents you are scoring on it.

Recommendation Engine: Collaborative Filtering Algorithm (user behavior based recommender) + Topic Modeling = Collaborative Topic Modeling
Using keywords (as content-based recommenders) over collaborative filtering to collaborative topic modeling recommenders.

Model Evaluation 
==============

- Human in the loop:
--- Topic Intrusion: first 3 topics in a document and the 4th is the least and human must distinguish.
--- Word Intrusion: first 10 words in a topic and the 11th is the least and human must distinguish.

- Held-out Log Likelihood metric: measure predictive power, not what the topics are, to differentiate between the models. But be careful about interpretability/precision of the model (Likelihood vs Interpretability trade-off: higher likelihood != higher interpretability). Interpretability measures by word and topic intrusions methods which measure human interpretability. So, higher interpretability for latent variables analysis and find the data story (inference), and higher likelihood for prediction prediction purposes like machine translation (word prediction)  and sentiment analysis (class prediction) applications using it as a language model. Don't forget to find the best number of k describes our topics in our corpus.

- Cosine Similarity Metric
--- interdistance = cosine similarity between 2 havles of document topic distribution.
--- intradistance = cosine similarity between 2 havles of 2 different document topic distributions.

- Size ( # of tokens assigned )

- Within-doc rank

- Similarity to corpus-wide distribution

- Locally-frequent words  

- Co-doc Coherence.

Gribb Sampling Algorithm:
For each iteration i:
     For each document d and word n currently assigned to z(old):
           Decrement n(d,z(old)) , v(z(old), w(d,n))
           Sample z(new) = k with probability proportional to conditional probability equation (gibbs sampling equation)
           Increment n(d,z(new)) , v(z(new), w(d,n))
