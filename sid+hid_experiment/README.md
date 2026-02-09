# H^2Rec

This is the implementation of H^2Rec framework.

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

You can pip install the `requirements.txt` to configure the environment.

## Preprocess the dataset

You can preprocess the dataset and get the LLMs embedding according to the following steps:

1. The raw dataset downloaded from website should be put into `/data/<yelp/fashion/beauty>/raw/`. The Yelp dataset can be obtained from [https://www.yelp.com/dataset](https://www.yelp.com/dataset). The fashion and beauty datasets can be obtained from [https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews).
2. Conduct the preprocessing code `data/data_process.py` to filter cold-start users and items. After the procedure, you will get the id file  `/data/<yelp/fashion/beauty>/hdanled/id_map.json` and the interaction file  `/data/<yelp/fashion/beauty>/handled/inter_seq.txt`.
3. Convert the interaction file to the format used in this repo by running `data/convert_inter.ipynb`.
4. To get the LLMs embedding for each dataset, please run the jupyter notebooks  `/data/<yelp/fashion/beauty>/get_item_embedding.ipynb` After the running, you will get the LLMs item embedding file `/data/<yelp/fashion/beauty>/handled/itm_emb_np.pkl`.
5. For hot start initialization, we need to run the jupyter notebook `data/pca.ipynb` to get the dimension-reduced LLMs item embedding for initialization, i.e., `/data/<yelp/fashion/beauty>/handled/pca64_itm_emb_np.pkl`.
6. For SID generation, please refer to the 'generate_semantic_codes_RQVAE.py' under '/data/yelp/handled' to generate the corresponding semantic code json file and embedding .pkl and .pth files.

After that we can run the main framework by setting your parameter using main.py.
# Organization of the framework
The whole structure of the framework are listed in the 'DualTrisRec.py' under the 'model' file.

The basic semantic codes embeddings are constructed in 'RQVAEEmbedding.py', also under the 'model' file.

Since we change the traditional '1 to 1' InfoNCE to '1 to many' with our positive sample selections, We precompute the positive samples using 'precompute_positive_pairs_v2.py' to accelerate the loss calculation.