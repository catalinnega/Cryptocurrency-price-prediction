import numpy as np
import sys
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import timedelta, date
import mylib_dataset as md

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)



if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

#def main():
#if(1):
def printTweet(descr, t):
    print(descr)
    print("date: %s\n" % t.date)
    print("text: %s\n" % t.text)
    print("retweets: %s\n" % t.retweets)


def sentiment_indicator_nltk(data, 
                             start_date = date(2014, 10, 2), 
                             end_date = date(2018, 11, 20),
                             key = 'Bitcoin',
                             max_tweets = 50):
    dates = []
    for single_date in daterange(start_date, end_date):
        dates.append(str(single_date))
        #print(single_date.strftime("%Y-%m-%d"))
    
    results = {}
    for i in range(len(dates)):
        results_tmp = {dates[i]: {'neg' : 0, 'pos' : 0}}
        results.update(results_tmp)
    
        
    for n in range(len(dates)-1):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(key).setSince(dates[n]).setUntil(dates[n+1]).setMaxTweets(max_tweets)
        tweet = got.manager.TweetManager.getTweets(tweetCriteria)
        
        sid = SentimentIntensityAnalyzer() 
        for i in range(len(tweet)):
          tweet2 = tweet[i]
          ss = sid.polarity_scores(tweet2.text)
          for k in sorted(ss):
             print('{0}: {1}, '.format(k, ss[k]), end='')
    
          results[dates[n]]['neg'] += ss['neg']
          results[dates[n]]['pos'] += ss['pos']

          printTweet("### Bitcoin tweets", tweet2)
               
    start_date = md.get_date_from_UTC_ms(data['dataset_dict']['UTC'][0])
    end_date = md.get_date_from_UTC_ms(data['dataset_dict']['UTC'][-1])
    
    start = False
    end = False
    sentiment_indicator_positive = np.zeros(len(data['data']))
    sentiment_indicator_negative = np.zeros(len(data['data']))
    step_1_day = 15*4*24
    for i in range(len(dates)):
        if(dates[i].find(end_date) != -1):
            end = True
        if(dates[i].find(start_date) != -1):
            start = True
        if(start and not end):
            ### offset with one step to adjust for real-time data feed.
            sentiment_indicator_positive[step_1_day * (i+1) : step_1_day * (i+2)] = results[dates[i]]['pos']
            sentiment_indicator_negative[step_1_day * (i+1) : step_1_day * (i+2)] = results[dates[i]]['neg']
    return np.array(sentiment_indicator_positive), np.array(sentiment_indicator_negative)

directory = '/home/catalin/databases/klines_2014-2018_15min/'
data = md.get_dataset_with_descriptors2(concatenate_datasets_preproc_flag = True, 
                                       preproc_constant = 0.99, 
                                       normalization_method = "rescale",
                                       dataset_directory = directory,
                                       hard_coded_file_number = 0,
                                       feature_names = [''])        
sentiment_indicator_positive, sentiment_indicator_negative = sentiment_indicator_nltk(data, 
                                                                                      start_date = date(2014, 10, 2), 
                                                                                      end_date = date(2018, 11, 20),
                                                                                      key = 'Bitcoin',
                                                                                      max_tweets = 50)



