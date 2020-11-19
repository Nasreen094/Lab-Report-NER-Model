Lab report entity extractor

Steps to run the service:

1. open the terminal from the base directory
2. run the command:   docker build -t lab_report_ner_image .   (this will create a docker image named lab_report_ner_image with some image id)
3. run the command: docker run -p 5000:5000 lab_report_ner_image    (this command will start running the python service in port number 5000)




You need to make sure that the previous container you launched is killed, before launching a new one that uses the same port. 
To list the container that is using port 5000, run the command below:

docker container ls

To kill the container that uses port 5000:

docker rm -f <container-name>


command to kill the existing process running in port 5000:    fuser -k 5000/tcp




