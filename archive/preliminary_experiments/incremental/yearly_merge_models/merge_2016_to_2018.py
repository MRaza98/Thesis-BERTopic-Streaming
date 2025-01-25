from bertopic import BERTopic

## Importing models from 2016 to 2018

topic_model_2016 = BERTopic.load("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_2016_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-04_00-40-19_efdd56")
topic_model_2017 = BERTopic.load("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_2017_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-04_00-46-39_fac649")
topic_model_2018 = BERTopic.load("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/model_2018_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-04_00-55-50_2faf61")

# Combine all models into one
merged_model = BERTopic.merge_models([topic_model_2016
                                    , topic_model_2017
                                    , topic_model_2018]
                                    , min_similarity=0.7)

# Save the merged model
merged_model.save("/home/raza/projects/Streaming-Pipeline-Parliamentary-Debates/models/merged_model_2016_to_2018_all_mpnet_base_v2_umap_hdbscan_speeches_Nonetopics_10dim_2024-11-15")

# Load the merged model
