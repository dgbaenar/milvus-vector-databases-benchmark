# Vector Database Benchmark

## Start Milvus with Docker

https://milvus.io/docs/install_standalone-docker.md


# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start the Docker container
$ bash standalone_embed.sh start



Stop and delete Milvus
You can stop and delete this container as follows

# Stop Milvus
$ bash standalone_embed.sh stop

# Delete Milvus data
$ bash standalone_embed.sh delete
