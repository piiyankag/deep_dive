# deep_dive


## Preprocessing
- Extract images from json files inside `raw_data/CleanSea`:
  - `python -m extract_images`
  - Images are saved in `preprocessed_data/original_sizes/<category>`

- Resize images to 512x512 and save them in `preprocessed_data/512x512/<resize method>/<category>`:
  - `python -m resize_images`


## Cloud Run
`gcloud run deploy --image $GCR_REGION/$GCP_PROJECT/$GCR_IMAGE:prod --memory $GCR_MEMORY --region $GCP_REGION --env-vars-file .env.yaml`
