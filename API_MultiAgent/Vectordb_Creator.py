# This is the Vectordb_Creator.py file
# It will handle the creation of the Milvus vector databases
# It is also going to have FastAPI integration so it can receive requests to create vector dbs
#
from pymilvus import MilvusClient, DataType, Collection
from fastapi import FastAPI


# FastAPI app
app = FastAPI()

# Milvus setup will be done in this class

# Connect to Milvus
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
    db_name="rag_db"
)

#Create database
if "rag_db" not in client.list_databases():
    client.create_database(db_name="rag_db")

#Define collection schema
schema = MilvusClient.create_schema(auto_id=True)

#I was thinking of having an endpoint here to receive the fields that will be used to create the collection
# But i dont think its necessary, vector databases have a pretty standard schema

#id field: INT64, primary key, auto_id
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)

#text field: STRING, max_length=65535
schema.add_field("text", DataType.VARCHAR, max_length=65535)

#metadata field: STRING, max_length=65535
schema.add_field("metadata", DataType.VARCHAR, max_length=65535)

#embedding field: FLOAT_VECTOR, dim=768
#384 is the dimension of the embeddings from HuggingFace sentence-transformers/all-MiniLM-L6-v2
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=384)

#Create collection
#Here we do need a FastAPI endpoint to receive the collection name
# We should have an if here to check if the collection already exists, but for now we will skip that

#Endpoint to receive collection name
@app.post("/create_collection/{collection_name}")
def create_collection(collection_name: str):
    # Check if collection already exists
    if collection_name in client.list_collections():
        return {"message": f"Collection {collection_name} already exists."}
    
    # Create collection with correct parameter names
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    return {"message": f"Collection {collection_name} created successfully."}

