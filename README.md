# Smart Student Performance Prediction — Live Demo

An interactive Streamlit prototype for exploring student data and demonstrating score prediction with a Random Forest regressor.

## Important transparency note
The included `sample_student_data.csv` is **synthetic illustrative data**. It exists only so the demo runs immediately. Its metrics are not validated research results and must not be represented as findings from real students. Replace it with your approved project dataset before reporting project outcomes.

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## Publish a public demo with Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `sample_student_data.csv`.
2. Open https://share.streamlit.io/ and sign in.
3. Choose **Create app**, select your repository, branch, and `app.py`.
4. Deploy. Streamlit will provide a public `*.streamlit.app` URL.
5. Open the public URL in a private/incognito browser window to confirm that it works without signing in.
6. Use that public app URL in the application form, not the GitHub repository URL.

## Suggested application description
“Interactive prototype for student performance and grade prediction, featuring CSV upload, exploratory visualizations, Random Forest regression, model evaluation, and individual score estimation.”

## Expected CSV format
The included CSV shows an example schema. Your actual CSV should contain numeric predictor columns and a numeric target column (for example, `final_score`). The app lets you select the target and features after upload.

## Privacy
Do not upload identifiable student records to a public demo. Use anonymized data and obtain required permissions.
