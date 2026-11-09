# DewDune: business viability dashboard

DewDune is a climate-smart, personalised skincare subscription for UAE residents. This Streamlit app tests whether the idea is sustainable. It covers:

- **Startup plan and questionnaire.** 31 questions, each mapped to at least one algorithm.
- **Synthetic survey.** 3,600 respondents (`generate_data.py`) with deliberate noise: duplicates, messy labels, impossible values, outliers, missing answers and straight-liners.
- **Cleaning and EDA.**
- **Classification.** Will this person subscribe? (Q30)
- **Regression.** How much will they pay? (Q31)
- **Clustering.** Customer segments and the first segment to target.
- **Association rules.** Box bundles. (Q28)
- **36-month financial model.** Scored against 7 sustainability tests, with sliders to change every assumption.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push this repo to GitHub. On share.streamlit.io, create an app, pick this repo and set the main file to `app.py`.
