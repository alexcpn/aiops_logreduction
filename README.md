# Log File reduction with TFIDF ranking and TimeSeries Fitting


***For Running the code please see [RUNME.md](RUNME.md)***

## Summary

Using TFIDF ( Term Frequency, Inverse Document Frequency) Vectorize to find the terms or logs that are least common
To explain the concept, lets take a snippet from a  modified sys log file [Log File Snippet][1]

Here we can see that certain logs occur only few times in the document and the majority of logs are periodic.
It could be that the problem is in the few logs that do not keep repeating ; example an IO driver that logs about possible disk error.
It could also be that there is a inherent problem in the system and the periodic logs are generated due to it. Practically we observe that many of the logs that repeats in a commercial non critical SW system can be ignored.

**Our goal is a system that can suppress the majority of periodically repeating logs ; but able to highlight the in-frequent logs.**

So we have two aspects here, the time pattern and the frequency of the logs itself. Let's take each of these terms.

## Log Frequency

We need to use a ranking where the most repeated similar logs have similar rankings and the least repeated logs are outliers.

We need a *similarity match* and not an *exact match* as logs that denote even the same event can be dissimilar in some numerical data between them. An example below

```
kubelet_volumes.go:154] Orphaned pod <aplha-numeric pod id> found, but volume paths are still present on disk : There were a total of 3 errors similar to this. Turn up verbosity to see them.
```

There are many text similarity algorithms for this https://stackoverflow.com/questions/17388213/find-the-similarity-metric-between-two-strings

Example could be the Jaccard Index, Cosine Similarity or the Jaro-Wrinler distance
For some of these the edit distance between two sequences to form the basis of the similarity algorithm; for others, we vectorise - that is convert each word in a sentence to a numerical representation and then use some similarity index of the resulting vectors to compare two sentence. The simplest being Cosine Similarity.

However these similarity algorithms are not very accurate.

We use the Scikit Learn Text Vectorize library and use the TfidfVectorizer. It first vectorize the words of the documents. Then it gives a higher rank to terms that are less repeating in the document over others.
  
A log sentence will have a lot of words. Many of which will be common. So once we vectorize each sentence, for each vectorized sentence, we select the word with the highest score as the representative score for the log sentence.
  
Illustration below
  
*Log rows*

 ```

Mar 31 09:31:48 1compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.659318010 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11855 (rc: 32)

Mar 31 09:31:48 2compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.751333697 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11856 (rc: 32)

3DDDDcompx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.888424524 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11864 (rc: 32)

Mar 31 09:31:48 4compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.783381048 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11857 (rc: 32)

Mar 31 09:31:48 5compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.826483871 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11858 (rc: 32)

Mar 31 09:31:49 compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.156622971 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11859 (rc: 32)

```

TFIDF based score for the log lines above (last column)

We can see that the only the last line has terms that are repeated most frequently in the entire document , and gets the lowest score. Whereas the other logs have terms like <n>compx-** which is unique in the entire document and gets the highest score

*TFID Score*

Snapshot here 

![TFIDF Rank](https://i.imgur.com/ZPcWfYL.png)  

Output - https://gist.github.com/alexcpn/07e40d4bb46397632f83ffdc0362e9bb#file-tfidfrank-csv

## Time Pattern

  Once we have the TFIDF rank, we plot this number against the log time. We use the Time series forecasting tool Prophet to fit the TFIDF rank against the time. Why we are doing this is that Prophet has an lower and upper threshold band in which it tries to fir the observations. This means that if there is a time pattern in the log; that is if similar logs are repeating in similar time interval, they are fitted inside the upper and lower band. Prophet uses this band to plot a trend for use in forecasting. However we can use the band to fit expected logs inside it; and only those logs that are outside the band we threshold out.

  

![Thresholding](https://i.imgur.com/mpu9jJ5.png)


*Final Sample Ouput*

Not for the graph above, which is a proper syslog file, while below is a small snippet of  syslog and is just illustrative in how 

Also here https://gist.github.com/alexcpn/07e40d4bb46397632f83ffdc0362e9bb#file-output-csv
```
Mar 31 09:31:48 2compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.751333697 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11856 (rc: 32)
Mar 31 09:31:48 3DDDDcompx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.888424524 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11864 (rc: 32)
Mar 31 09:31:48 4compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.783381048 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11857 (rc: 32)
Mar 31 09:31:48 5compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.826483871 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11858 (rc: 32)
Mar 31 09:31:49 compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.441979340 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11860 (rc: 32)
Mar 31 09:31:49 compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.501112749 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11861 (rc: 32)

```
  
## Reference

[1]: https://gist.github.com/alexcpn/07e40d4bb46397632f83ffdc0362e9bb#file-input_ssylog-csv

Log File Snippet

```
Mar 31 09:31:48 1compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.659318010 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11855 (rc: 32)

Mar 31 09:31:48 2compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.751333697 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11856 (rc: 32)

3DDDDcompx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.888424524 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11864 (rc: 32)

Mar 31 09:31:48 4compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.783381048 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11857 (rc: 32)

Mar 31 09:31:48 5compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:48.826483871 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11858 (rc: 32)

Mar 31 09:31:49 compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.156622971 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11859 (rc: 32)

Mar 31 09:31:49 compx-f476c9ffb-8gtc4 ns-slapd[2762]: [31/Mar/2022:09:31:49.441979340 +0000] DSRetroclPlugin - delete_changerecord: could not delete change record 11860 (rc: 32)

..The above lines repeated ...

```

