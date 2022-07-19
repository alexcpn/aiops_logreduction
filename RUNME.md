# Running Log Analysis


# Read a log file and reduce via TFIDF -> Preffered

```
docker run --rm -it --net=host -v /home/alex:/home alexcpn/sklearn_prophet_python:1

Ex 
cd /home/coding/anomalydetection
python ./python/loganalysis/logoutlier_tfidf.py /home/Downloads/logs/1000_dc1-p1-r3-srv-2-journalctl_error.txt ./out/

Open Browser at http://127.0.0.1:8050/ to see GUI

```

For testing

```
python ./python/loganalysis/logoutlier_full.py ./python/resources/test1.log ./out/err.log
```

## Read a log file and reduce it via Spacy

```
docker run --rm -it --net=host -v /home/alex:/home /alexcpn/fb_prophet_python:1

Load the bigger model if neeeded
python -m spacy download en_core_web_md


cd /home/coding/anomalydetection

root@pop-os:/home/coding/anomalydetection# python ./python/loganalysis/logoutlier_spacy.py <syslog path> <out path>
```
