cd chat-interface
docker build -t chat-interface .
cd ..
cd generation-server
docker build --no-cache -t generation-server:v1 .
cd ..