import requests
import pandas as pd
import json
from pyyoutube import Api
import os
import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_REGISTRY_URI"] = "/home/akashy/staidg/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

key = "AIzaSyBH-O7CsoPOrtlgI9ZMu-mF6-PaLjXpabE"
api = Api(api_key=key)

query = "'Data Science'"
video = api.search_by_keywords(q=query, search_type=["video"], count=100, limit=300)

maxResults = 10000
nextPageToken = ""
s = 0

with mlflow.start_run():
    dic={}
    for id_ in [x.id.videoId for x in video.items]:
        uri = "https://www.googleapis.com/youtube/v3/commentThreads?" + \
                "key={}&textFormat=plainText&" + \
                "part=snippet&" + \
                "videoId={}&" + \
                "maxResults={}&" + \
                "pageToken={}"
        uri = uri.format(key, id_, maxResults, nextPageToken)
        content = requests.get(uri).text
        data = json.loads(content)
        c=0
        for item in data['items']:
            c+=1
            if item['snippet']['topLevelComment']["snippet"]["publishedAt"] in dic.keys():
                dic[item['snippet']['topLevelComment']["snippet"]["publishedAt"]] += item['snippet']['topLevelComment']['snippet']['likeCount']
            else:
                dic[item['snippet']['topLevelComment']["snippet"]["publishedAt"]] = item['snippet']['topLevelComment']['snippet']['likeCount']

    df = pd.DataFrame.from_dict(dic, orient='index')
    df = df.reset_index()
    df.rename(columns = {0:'counts', 'index':'index'}, inplace = True )

    mlflow.log_artifact(local_path="/home/akashy/staidg//scripts/get_data.py",
                        artifact_path="get_data code")
    mlflow.end_run()

df.to_csv('/home/akashy/staidg/datasets/data.csv')