# Thesis Title
~~Near Real-Time~~ Multi-National Topic Modeling for Parliamentary Debates: A Streaming Pipeline Approach

# Expected Results
* Top 10 Topics Table for England, Spain, and Germany along with c_v scores
* Top 10 Topics Line Chart showing evolution over time
~~* Topic models evaluated for a chosen number of parliaments~~
  ~~* How does different batch size streamed affect the topic evaluation scores?~~
* A comparison of batch and online topic models
* Within the online approach, a comparison of different languages under an evaluation scheme
* Within the online approach, a comparison of recent and past topics

# Pipeline Building Blocks

* Data Stream => I decided to work with Kafka for learning purposes.
  * Message queuing protocol is sufficient or do we need Kafka?
    * Even asyncio is enough, but Kafka is a learning opportunity. Maybe I can justify it anyhow at the end.
* Data Processing
  * Is pre-processing necessary in the case of embeddings?
    * Not according to the author:
      * https://maartengr.github.io/BERTopic/faq.html#how-do-i-remove-stop-words
      * https://maartengr.github.io/BERTopic/faq.html#should-i-preprocess-the-data
    * However, there is a stopwords removal step with CountVectorizer:
      * from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer_model = CountVectorizer(stop_words="english")
        topic_model = BERTopic(vectorizer_model=vectorizer_model)
    * We can also use the ClassTfidfTransformer to reduce the impact of frequent words:
      * from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        topic_model = BERTopic(ctfidf_model=ctfidf_model)

  * Which embedding model is best for our use case and why?
    * paraphrase-multilingual-MiniLM-L12-v2 and paraphrase-multilingual-mpnet-base-v2 are recommended: https://maartengr.github.io/BERTopic/faq.html#which-embedding-model- should-i-choose


* Backend
  * Postgres database
* Frontend

* Notes:
   * Results not being consistent between runs: https://maartengr.github.io/BERTopic/faq.html#why-are-the-results-not-consistent-between-runs
   * 
