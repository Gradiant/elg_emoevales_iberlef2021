# elg_emoevales_iberlef2021
elg_emoevales_iberlef2021 is a tool that applies sentiment analysis classifitacion over texts.
This repository contains a dockerized API built over emoevales_iberlef2021 for integrate it into the ELG. Its original code can
be found [here](https://github.com/gsi-upm/emoevales-iberlef2021).

## Install

```
sh docker-build.sh
```

## Execute
```
docker run --rm -p 0.0.0.0:8866:8866 --name emoeval elg_emoevales_iberlef2021:1.0
```
## Use

```
curl -X POST  http://0.0.0.0:8866/predict_json -H 'Content-Type: application/json' -d '{"type": "text", "content":"El cabrón del árbitro nos ha fastidiado el partido con ese penalti injusto que ha pitado"}'
```

# Test
In the folder `test` you have the files for testing the API according to the ELG specifications.
It uses an API that acts as a proxy with your dockerized API that checks both the requests and the responses.
For this follow the instructions:
1) Configure the .env file with the data of the image and your API
2) Launch the test: `docker-compose up`
3) Make the requests, instead of to your API's endpoint, to the test's endpoint:
   ```
   curl -X POST  http://0.0.0.0:8866/processText/service -H 'Content-Type: application/json' -d '{"type": "text", "content":"El cabrón del árbitro nos ha fastidiado el partido con ese penalti injusto que ha pitado"}'
   ```
4) If your request and the API's response is compliance with the ELG API, you will receive the response.
   1) If the request is incorrect: Probably you will don't have a response and the test tool will not show any message in logs.
   2) If the response is incorrect: You will see in the logs that the request is proxied to your API, that it answers, but the test tool does not accept that response. You must analyze the logs.


## Citations
The original work of this tool is the following:
 - Iglesias, C. A. (2021). GSI-UPM at IberLEF2021: Emotion Analysis of Spanish Tweets by Fine-tuning the XLM-RoBERTa Language Model.
 - https://github.com/gsi-upm/emoevales-iberlef2021
