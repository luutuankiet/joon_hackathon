```shell
gcloud firestore indexes composite create --project=joon-sandbox --database="joon-hackathon-chatbot" --collection-group=confluence 
--query-scope=COLLECTION --field-config=vector-config='{"dimension":"768","flat": "{}"}',field-path=embedding
```
