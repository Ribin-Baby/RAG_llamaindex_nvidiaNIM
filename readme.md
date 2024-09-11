# RAG application using LLamaIndex and NVIDIA NIM microservice
* Here we are usingLlamaIndex to interact with NVIDIA hosted NIM microservices like chat, embedding, and reranking models to build a simple retrieval-augmented generation (RAG) application. 

# How to use ?
```sh
 curl -X 'GET' 'http://127.0.0.1:8000/query' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"query": "where is sweden located?"}'
 ```

# How to containarize ?

* build the docker image 
> docker build -t chatapp .
* `chatapp` is the image name

* run the container from the above created image
> docker run -p 8000:8000 chatapp

* this will expose the REST API to the application, which can be accessed from the host machine at the following endpoint
> http://localhost:8000/query