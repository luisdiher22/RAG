#This class will handle loading information to the vector db desired by the user
# it will have FastAPI where we will receive the collection name and the data to be loaded
# We can implement different data loaders here depending on the file type
from pymilvus import MilvusClient, Collection, connections
from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI app
app = FastAPI()

class DataItem(BaseModel):
    text: str
    metadata: str
    embedding: list[float]

class LoadDataRequest(BaseModel):
    data: list[DataItem]

# Connect to Milvus - establecer conexión global
MILVUS_URI = "localhost"
MILVUS_PORT = "19530"
MILVUS_TOKEN = "root:Milvus"
DATABASE_NAME = "rag_db"

# Establecer conexión para Collection API
connections.connect(
    alias="default",
    host=MILVUS_URI,
    port=MILVUS_PORT,
    user="root",
    password="Milvus",
    db_name=DATABASE_NAME
)

# Client para operaciones generales
client = MilvusClient(
    uri=f"http://{MILVUS_URI}:{MILVUS_PORT}", 
    token=MILVUS_TOKEN,
    db_name=DATABASE_NAME
)

# Lets receive the collection name and data to be loaded via a POST request
@app.post("/load_data/{collection_name}")
def load_data(collection_name: str, payload: LoadDataRequest):
    # Get the collection - ahora con la conexión en la DB correcta
    collection = Collection(name=collection_name, using="default")

    # Insert data
    # Data should be a list of dictionaries with keys: text, metadata, embedding
    texts = []
    metadatas = []
    vectors = []
    
    for item in payload.data:
        texts.append(item.text)
        metadatas.append(item.metadata)
        vectors.append(item.embedding)

    entities = [texts, metadatas, vectors]

    res = collection.insert(entities)

    #Now we need to create the index after loading the data
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="vector_index",
        params={"nlist": 128}
    )
    
    # Check if index exists before creating
    existing_indexes = [idx.field_name for idx in collection.indexes]
    if "embedding" not in existing_indexes:
        client.create_index(collection_name=collection_name, index_params=index_params)
    
    # Load the collection to make it ready for search
    collection.load()
    collection.flush()

    stats = client.get_collection_stats(collection_name=collection_name)
    count = int(stats["row_count"])
    print(f" Total entities in collection: {count}")

    return {"message": f"Data loaded successfully into collection {collection_name}. Inserted {res.insert_count} records"}



