# Topic-Modeling-using-Python
Using Topic Models in Latent-Variables Cluster-Analysis, Information Retrieval (search), Recommender System applications.

Topic models with SVD matrix factorization which used in 'Latent Semantic Analysis (LSA)' Algorithm are popular in Information Retrieval.

Topic models with a generative probabilistic model, a bagofwords model, Latent Dirichlet Allocation (LDA) Algorithm are popular in Latent Variables/Topics and structure Discovering (Semantic Analysis) or Topics/Words distributions Discovering. And in soft/fuzzy clustering whether in word clusters represents the topics as multinomial distribution of words, or in topic cluster represents the document as multinomial distribution of topics.

Each document gets represented as a pattern of LDA topics (multinomial distribution). Making Every document appear different enough to be separable, similar enough to be grouped, like fingerprint or DNA.

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

K determination:
1- Try different k, choose the best likelihood.
2- Use HDP-LDA, take uniform sample from corpus if large.
3- Back to 1, try small interval around k.

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

Model Evaluation 
==============

- Human in the loop:
--- Topic Intrusion: first 3 topics in a document and the 4th is the least and human must distinguish.
--- Word Intrusion: first 10 words in a topic and the 11th is the least and human must distinguish.

- Held-out Log Likelihood metric: measure predictive power, not what the topics are, to differentiate between the models. But be careful about interpretability/precision of the model (Likelihood vs Interpretability trade-off: higher likelihood != higher interpretability). Interpretability measures by word and topic intrusions methods which measure human interpretability. So, higher interpretability for latent variables analysis and find the data story (inference), and higher likelihood for prediction purposes like machine translation (word prediction)  and sentiment analysis (class prediction) applications using it as a language model. Don't forget to find the best number of k describes our topics in our corpus.

- Cosine Similarity Metric
--- interdistance = cosine similarity between 2 havles of document topic distribution.
--- intradistance = cosine similarity between 2 havles of 2 different document topic distributions.

- Size ( # of tokens assigned )

- Within-doc rank

- Similarity to corpus-wide distribution

- Locally-frequent words  

- Co-doc Coherence

Gribb Sampling Algorithm:
For each iteration i:
     For each document d and word n currently assigned to z(old):
           Decrement n(d,z(old)) , v(z(old), w(d,n))
           Sample z(new) = k with probability proportional to conditional probability equation (gibbs sampling equation)
           Increment n(d,z(new)) , v(z(new), w(d,n))

Why is LDA a mixture model?

LDA is not a mixture model. It is an admixture model or a mixed-membership model.
Mixture models have a single latent variable that denotes which cluster they're in. This is often written as an indicator variable z.
LDA is a model over documents (a bag of words), and has a latent variable for topic assignments for every token: z1…zNz1…zN
Thus, words can belong to different clusters. This intuitively makes sense because documents can be about more than one thing. I.e., about both technology and business. This often results in better models of real text than pure mixture models.

In a mixed-membership latent variable model, the definition of word co-occurrence is stricter that in a mixture model where only one topic is allocated per document. Words have to co-occur more tightly in a mixed-membership model. In other words, in a mixture model we assert that words in a document co-occur and given that we ask the model to return the highest probable topic. Whereas in a mixed-membership model we assert that words co-occur in a topic within a document and we ask the model to return the highest probable topics in that document. the Latent Dirichlet Allocation model is an example of a mixed-membership model.

What is the difference between LDA (linear analysis discriminant) and LDA (latent Dirichlet allocation)?

Linear Discriminant Analysis : LDA attempts to find a feature subspace that maximizes class separability. It is basically about supervised technique, which is primarily used for classification. LDA makes assumptions about normally distributed classes and equal class co-variances, however, this only applies for LDA as classifier and LDA for dimensional reduction can also work reasonably well if those assumptions are violated.
As LDA is supervised, so it needs labelled data. This draws various similarities between PCA too in term of the way it uses co-variance matrix.

Latent Dirichlet Allocation : is primarily an unsupervised technique used for finding topic distribution. It is based on Dirichlet Allocation and works on based of iteration to find best topic underlying solving using “Joint probabilities”. It is generally used on text data.

What is the difference between latent Dirichlet allocation and probabilistic topic models?

There are many flavors of probabilistic topic models. LDA is a prototypical example. pLSI is a precursor of LDA and correlated topic models are a successor. There are a cottage industry of other probabilistic topic models. That add additional features to topic models:

hierarchical LDA

LDAWN

Syntactic Topic Models

polylingual LDA

shLDA

supervised LDA

and many others ...


Is it still necessary to learn LDA (latent Dirichlet distribution)?

Necessary?
Well, I usually like to call the general area topic models, which itself is an instance of the general area of discrete matrix factorisation (like gamma-Poisson models, GaP). So in that sense, its much broader than simple old LDA.
But when teaching things like LDA and GaP, which should be taught together, the thing I like is the fact that you can teach a whole bunch of things in one go:

variational methods

Gibbs sampling

collapsed Gibbs sampling

hyper-parameter optimisation

problems with maximum likelihood methods

regularisation versus Bayesian MAP versus Bayesian MCMC

why we want non-parametrics (just the why… teaching NP comes later)

Its just a fabulous problem to explain a whole bunch of ideas on. So you might not need LDA, but matrix factorisation and all the above stuff are pretty handy to know. You’ll learn LDA as a by-product.



Why is LDA a mixture model?

LDA is not a mixture model. It's an admixture model or a mixed-membership model.

Mixture models have a single latent variable that denotes which cluster they're in. This is often written as an indicator variable z.

LDA is a model over documents (a bag of words), and has a latent variable for topic assignments for every token: Z1,...,Zn.

Thus, words can belong to different clusters. This intutively makes sense because documents cen be about more than one thing, I.e., about both technology and business. This often results in better models of real text than pure mixture models.


LDA topic model uses posterior/probabilistic inference to figure out what are the missing pieces of story (latent variables) that best explain our data, through:
- Topics to word types _ multinomial distribution.
- Documents to topics _ multinomial distribution.
So, 
Model = data story 
Topics = latent variables = missing pieces of story 
Statistical Inference = filling in those missing pieces

Generative Topic Models are language models, so it can be used in NLP applications.


LDA topic model uses posterior/probabilistic inference to figure out what are the missing pieces of story (latent variables) that best explain our data, through:
- Topics to word types _ multinomial distribution.
- Documents to topics _ multinomial distribution.
So, 
Model = data story 
Topics = latent variables = missing pieces of story 
Statistical Inference = filling in those missing pieces

Generative Topic Models are language models, so it can be used in NLP applications.

Topic Models are documents soft-clusters representing them as multinomial distribution of topics. Every topic represents as multinomial distribution of words.
We can see them also as Dimensionality Reduction models, which transform our documents from word-vector to topic-vector.

Recommendation Engine: Collaborative Filtering Algorithm (user behavior based recommender) + Topic Modeling = Collaborative Topic Modeling

Using keywords (as content-based recommenders) over collaborative filtering to collaborative topic modeling recommenders.

LDA Algorithm (Iterative Algorithm)
1. Initialize parameters.
2. Initialize topic assignments randomly.
3. Iterate: 
                For each word in each document:
                       Resample topic for word, giving all other words and their current topic assignments.
4. Get results.
5. Evaluate model.


Topic models with SVD matrix factorization which used in 'Latent Semantic Analysis (LSA)' Algorithm are popular in Information Retrieval.

Topic models with a generative probabilistic model, a bagofwords model, Latent Dirichlet Allocation (LDA) Algorithm are popular in Latent Variables/Topics and structure Discovering (Semantic Analysis) or Topics/Words distributions Discovering. And in soft/fuzzy clustering whether in word clusters represents the topics as multinomial distribution of words, or in topic cluster represents the document as multinomial distribution of topics.

LDA = probabilistic version of LSA = a fully Bayesian version of pLSI
LSA = PCA applied to text data = linear algebra method = LSI Latent Semantic Indexing
Topic models with SVD matrix factorization which used in 'Latent Semantic Analysis (LSA)' Algorithm are popular in Information Retrieval.

Topic models with a generative probabilistic model, a bagofwords model, Latent Dirichlet Allocation (LDA) Algorithm are popular in Latent Topics Discovering (Semantic Analysis).or Topics/Words distributions Discovering.

LDA = probabilistic version of LSA = a fully Bayesian version of pLSI
LSA = PCA applied to text data = linear algebra method = LSI Latent Semantic Indexing
