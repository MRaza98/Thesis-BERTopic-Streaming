from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import torch
torch.cuda.set_device(2)

# Initialize models with identical parameters to your setup
embedding_model = SentenceTransformer("all-miniLM-L6-v2")
umap_model = UMAP(
    n_components=8,
    n_neighbors=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

# Create distinct document sets
docs_2018 = [
    # Environmental and Climate documents
    "Climate change is a pressing issue requiring immediate action by all nations",
    "Environmental protection measures need to be strengthened globally",
    "Renewable energy sources are crucial for sustainability in modern times",
    "Carbon emissions must be reduced significantly to meet climate goals",
    "Green technology investments are essential for future growth",
    "Solar power implementation has shown promising results worldwide",
    "Wind farms are becoming increasingly efficient energy sources",
    "Ocean pollution threatens marine ecosystems dramatically",
    "Forest conservation plays a vital role in biodiversity",
    "Sustainable agriculture practices are gaining momentum",
    "Electric vehicles are revolutionizing transportation sector",
    "Waste management systems need urgent modernization",
    "Biodiversity loss threatens ecological balance severely",
    "Urban planning must incorporate environmental considerations",
    "Clean water access remains a global challenge",
    "Recycling programs require better implementation strategies",
    "Air quality improvements demand stricter regulations",
    "Deforestation impacts climate change significantly",
    "Renewable energy storage solutions are advancing rapidly",
    "Sustainable building practices are becoming standard",
    
    # Economic and Infrastructure documents
    "Infrastructure development needs substantial investment",
    "Economic growth must be balanced with sustainability",
    "Public transportation systems require modernization",
    "Digital infrastructure is crucial for economic development",
    "Urban development faces new challenges globally",
    "Rural infrastructure needs significant improvements",
    "Smart city initiatives are gaining popularity worldwide",
    "Transportation networks require extensive upgrades",
    "Building regulations need stricter enforcement",
    "Economic policies must address environmental concerns",
    "Infrastructure maintenance costs are rising annually",
    "Public works projects face funding challenges",
    "Urban planning requires innovative solutions",
    "Construction standards are evolving rapidly",
    "Infrastructure security needs greater attention"
]

docs_2019 = [
    # Healthcare documents
    "Healthcare reform is needed to improve access for all",
    "Medical research funding should be increased substantially",
    "Public health initiatives need more support and resources",
    "Hospital capacity needs to be expanded in rural areas",
    "Mental health services require immediate attention",
    "Telemedicine is revolutionizing healthcare delivery",
    "Preventive healthcare measures save costs long-term",
    "Healthcare worker shortage needs urgent attention",
    "Medical technology advances improve patient care",
    "Healthcare accessibility remains a major challenge",
    "Mental health awareness programs need expansion",
    "Emergency medical services require better funding",
    "Healthcare infrastructure needs modernization",
    "Patient care quality standards must be improved",
    "Medical education requires curriculum updates",
    "Healthcare data security needs strengthening",
    "Elderly care facilities need better resources",
    "Public health education requires more funding",
    "Healthcare insurance systems need reform",
    "Medical research breakthroughs show promise",
    
    # Education and Technology documents
    "Digital education platforms are transforming learning",
    "Technology integration in classrooms is essential",
    "Educational resources need better distribution",
    "Online learning platforms show promising results",
    "Student assessment methods need modernization",
    "Teacher training programs require updates",
    "Educational technology investment is crucial",
    "School infrastructure needs improvement",
    "Distance learning programs are expanding rapidly",
    "Education accessibility remains challenging",
    "STEM education requires more resources",
    "Student support services need enhancement",
    "Educational equity needs greater attention",
    "School safety measures require updating",
    "Special education programs need more funding"
]

# Train both models separately
model_2018 = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    verbose=True
)
model_2019 = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    verbose=True
)

# Fit both models
topics_2018, _ = model_2018.fit_transform(docs_2018)
topics_2019, _ = model_2019.fit_transform(docs_2019)

print("\n2018 Model Topics:")
print(model_2018.get_topic_info())
print("\nTop words for each 2018 topic:")
for topic in model_2018.get_topics():
    if topic != -1:  # Skip outlier topic
        print(f"Topic {topic}: {model_2018.get_topic(topic)[:5]}")

print("\n2019 Model Topics:")
print(model_2019.get_topic_info())
print("\nTop words for each 2019 topic:")
for topic in model_2019.get_topics():
    if topic != -1:  # Skip outlier topic
        print(f"Topic {topic}: {model_2019.get_topic(topic)[:5]}")

# Merge both models simultaneously (like in the documentation)
merged_model = BERTopic.merge_models([model_2018, model_2019])

print("\nMerged Model Topics:")
print(merged_model.get_topic_info())
print("\nTop words for each merged topic:")
for topic in merged_model.get_topics():
    if topic != -1:  # Skip outlier topic
        print(f"Topic {topic}: {merged_model.get_topic(topic)[:5]}")

# Check topic similarities with both original models
print("\nTopic Similarities with 2018 model:")
for topic_2018 in model_2018.get_topics():
    if topic_2018 != -1:
        similarities = []
        for topic_merged in merged_model.get_topics():
            if topic_merged != -1:
                words_2018 = set(word for word, _ in model_2018.get_topic(topic_2018)[:10])
                words_merged = set(word for word, _ in merged_model.get_topic(topic_merged)[:10])
                similarity = len(words_2018.intersection(words_merged)) / len(words_2018.union(words_merged))
                similarities.append((topic_merged, similarity))
        print(f"2018 Topic {topic_2018} similarities: {sorted(similarities, key=lambda x: x[1], reverse=True)[:3]}")

print("\nTopic Similarities with 2019 model:")
for topic_2019 in model_2019.get_topics():
    if topic_2019 != -1:
        similarities = []
        for topic_merged in merged_model.get_topics():
            if topic_merged != -1:
                words_2019 = set(word for word, _ in model_2019.get_topic(topic_2019)[:10])
                words_merged = set(word for word, _ in merged_model.get_topic(topic_merged)[:10])
                similarity = len(words_2019.intersection(words_merged)) / len(words_2019.union(words_merged))
                similarities.append((topic_merged, similarity))
        print(f"2019 Topic {topic_2019} similarities: {sorted(similarities, key=lambda x: x[1], reverse=True)[:3]}")