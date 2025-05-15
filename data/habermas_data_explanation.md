# Habermas Machine Dataset Explanation

## Available Datasets

1. **Candidate Comparisons** (`hm_all_candidate_comparisons.parquet`):
   - A large dataset (39,955 rows × 104 columns) containing comparisons between different candidate statements
   - Contains detailed metadata about candidates and their rankings

2. **Final Preference Rankings** (`hm_all_final_preference_rankings.parquet`):
   - A dataset (10,455 rows × 12 columns) with numerical rankings and metadata
   - Contains timestamp and worker ID information

3. **Position Statement Ratings** (`hm_all_position_statement_ratings.parquet`):
   - A dataset (45,976 rows × 26 columns) with ratings for different position statements
   - Contains affirming and negating statements for different topics/questions

4. **Round Survey Responses** (`hm_all_round_survey_responses.parquet`):
   - A dataset (21,608 rows × 51 columns) with detailed survey responses
   - Contains the issue/question, individual opinions, and consensus statements

## Key Data Structure for (Issue, Opinions, Consensus Statement)

The Round Survey Responses dataset is the most relevant for extracting the structure you're looking for:

- **Issue**: Found in the `question.text` column
  - 2,263 unique questions across 105 unique topics
  - Example: "Should parents be allowed to opt out of sex education for their children?"

- **Opinions**: Found in the `opinion.text` column
  - Individual opinions about the issue
  - Example: "I think sex education should be somewhat mandatory in the schools..."

- **Consensus Statement**: Found in the `candidate.text` column
  - The consensus statement that attempts to represent collective viewpoints
  - Example: "Sex education is very important in schools, as it is not always something that is spoken about at home..."

## Sample Code to Extract Data

```python
import io
import requests
import pandas as pd

# Get the dataset
file_location = 'https://storage.googleapis.com/habermas_machine/datasets/hm_all_round_survey_responses.parquet'
response = requests.get(file_location)
df = pd.read_parquet(io.BytesIO(response.content))

# Create a dataset with just the required columns
structured_df = df[['question.text', 'opinion.text', 'candidate.text']].rename(
    columns={
        'question.text': 'issue', 
        'opinion.text': 'opinion', 
        'candidate.text': 'consensus_statement'
    }
)

# Filter as needed and use for analysis
# For example, to get all data for a specific issue:
specific_issue = "Should parents be allowed to opt out of sex education for their children?"
issue_data = structured_df[structured_df['issue'] == specific_issue]

# To get unique issues:
unique_issues = structured_df['issue'].unique()
```