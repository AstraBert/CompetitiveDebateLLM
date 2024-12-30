# DebateLLM Championship

### Tournament structure

> _5 LLMs, 1vs1 matches to produce the most convincing argumentation in favor or against a random motion. Oh, and also the debate judge is an LLM :)_

The tournament is structured with the so called "Italian" formula, meaning that all participants play with all the others. There is no "home and away games" schema: every participant plays with each of the other ones only once. A model earns one point by winning a game, whereas it does not earn any (but it does not lose any as well) when losing a game.  

Each tournament round is one-shot, meaning that each participant has only one possibility to generate a 150-250 words argument, that will be then judged by an external LLM.

This first tournament consists of 5 LLMs as *debaters*:

- [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [`Qwen/Qwen2.5-72B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
- [`microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- [`HuggingFaceH4/starchat2-15b-v0.1`](https://huggingface.co/HuggingFaceH4/starchat2-15b-v0.1)
- [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

And two as *judges*:

- [`Qwen/QwQ-32B-Preview`](https://huggingface.co/Qwen/QwQ-32B-Preview)
- [`meta-llama/Llama-3.3-70B-Instruct`](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

### Data Collection and Processing

> _Code reference: [DebateChampionshipLLMs.ipynb](https://github.com/AstraBert/DebateLLM-Championship/blob/main/DebateChampionshipLLMs.ipynb)_

The motions which were used to prompt the debate matches were extracted from [`kokhayas/english-debate-motions-utds`](https://huggingface.co/datasets/kokhayas/english-debate-motions-utds) dataset on HuggingFace. 

1,000 of them were then randomly sampled from the 10,000+ set of motions contained in the original dataset, and a random motion was selected for each debate round.

```python
from datasets import load_dataset

# download the dataset from HF hub
dts = load_dataset("kokhayas/english-debate-motions-utds")
dtsdct = dts["train"]
     
import random as r

# sample 1000 motions from the original dataset
motions = dtsdct["motion"]
motions2use = []
numbers = []
j = 0
while j < 1000:
    n = r.randint(0,10000)
    if n not in numbers:
        numbers.append(n)
        if motions[n].lower().startswith("th"):
            motions2use.append(motions[n])
            j+=1
        else:
            continue
    else:
        continue
```

### Building and Running the Tournament

> _Code reference: [DebateChampionshipLLMs.ipynb](https://github.com/AstraBert/DebateLLM-Championship/blob/main/DebateChampionshipLLMs.ipynb)_

We approached building the tournament by:

- decomposing it into its atomic parts, the "building blocks" (defining how debaters and judges generate their answers)
- scaling to creating the structure of one round (debater 1 -> debater 2 -> judge)
- defining the entire tournament as a loop of rounds, with debate data collection and points tracking (for the final ranking) 

The code to create the building blocks of the debate tournament is the following:

```python
from huggingface_hub import InferenceClient
from google.colab import userdata

# create an HF client for inference
hf_token = userdata.get('HF_TOKEN_INFERENCE')
client = InferenceClient(api_key=hf_token)

# define a function for the debaters to produce their argument
def debate_inference(model, prompt):
  messages = [
	  {"role": "system", "content": "You are skilled in competitive debate. You produce arguments that strictly adhere to the position you are required to take by the prompts you are proposed with"},
	  {"role": "user", "content": prompt}
  ]
  completion = client.chat.completions.create(
    model=model,
  	messages=messages,
  	temperature=0.5,
  	max_tokens=2048,
  	top_p=0.7
  )
  return completion.choices[0].message.content

# define a function for the judges to produce their verdict
def judge_inference(model, motion, essay1, essay2):
  messages = [
	  {"role": "system", "content": "You are a judge, based on the motion, the argumentation in favor of it and the argumentation against it, you should produce a JSON string that contains the following fields:\n\n- winner (str): can take only FAVOR or AGAINST as values, based on who you think the winner is\n- reasons (str): the reasons why you chose the winner. OUTPUT ONLY THE JSON STRING AS: '''\n\n```json\n{\"winner\": 'FAVOR'|'AGAINST', \"reasons\": 'Reasons why you chose the winner'}\n```\n\n'''"},
	  {"role": "user", "content": "MOTION:\n"+motion},
	  {"role": "user", "content": "ARGUMENT IN FAVOR:\n"+essay1},
	  {"role": "user", "content": "ARGUMENT AGAINST:\n"+essay2},
    {"role": "user", "content": "Who is the winner? OUTPUT ONLY THE JSON STRING AS: '''\n\n```json\n{\"winner\": 'FAVOR'|'AGAINST', \"reasons\": 'Reasons why you chose the winner'}\n```\n\n'''"}
  ]
  completion = client.chat.completions.create(
    model=model,
  	messages=messages,
  	temperature=0,
  	max_tokens=2048,
  	top_p=0.7
  )
  return completion.choices[0].message.content

# define a tournament round
def tournament_round(model1, model2, judge, motion):
  prompt1 = "Produce an essay of maximum 150 words in favor of this motion: " + motion
  prompt2 = "Produce an essay of maximum 150 words against this motion: " + motion
  essay1 = debate_inference(model1, prompt1)
  essay2 = debate_inference(model2, prompt2)
  winner_answer = judge_inference(judge, motion, essay1, essay2)
  return essay1, essay2, winner_answer
```

For the tournament itself to be run, we add the following features to the backbone structure:

- Point tracking
- Debate data collection
- *winner* and *reasons for winner's choice* extraction from the judge's answer

The last point is especially painful, since the judge's answer can come in various formats even if the system instructions are very clear on how to structure it, so we decided to tackle the challenge posed by the variability of the output by adding a *output parser* LLM. This output parser LLM is `gpt-4o-mini`, that is wrapped into Langchain OpenAI chat class (`ChatOpenaAI`), and linked to a Pydantic schema for structured output generation:

```python
from google.colab import userdata
import os

# set OpenAI API key as an environment variable
a = userdata.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = a

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# generate a chat prompt template with Langchain, to wrap your system instructions for the model
GPT_MODEL = "gpt-4o-mini"
llm = ChatOpenAI(temperature=0, model=GPT_MODEL)
system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Your job is to restructure the virdict from a debate competition so that it follows this structure:
            - winner: the winner, as reported by the virdict
            - reasons: reasons for the choice of the winner
            Strictly follow the virdict you are provided with, do not add/make up any information."""),
        ("human", "{message}"),
    ]
)

from pydantic import BaseModel, Field

# create a Pydantic BaseModel for structured output generation
class Verdict(BaseModel):
    """Structure of the output of a debate competition verdict"""
    winner: str = Field(description="The winner, as reported by the verdict")
    reasons: str = Field(description="Reasons for the choice of the winner")

# define an inference-ready system instructions+LLM+structured output parser 
chain = system_prompt | llm.with_structured_output(Verdict)
```

Now we can run the tournament:

```python
import time

# define points tracker
modelpoints = {judges[i]: {model: 0 for model in models} for i in range(len(judges))}

# define data collector
motions2args2winner2reasons = {"motions": [], "judge": [], "favor_model": [], "favor_arg": [], "against_model": [], "against_arg": [], "winner": [], "reasons": [], "total_time": []}

judge_counter = 0
for judge in judges:
  judge_counter+=1
  pairs = []
  counter = 0
  for i in range(len(models)):
    for j in range(len(models)):
      # only make two models play with each other if they have not met before
      if i!=j and (i,j) not in pairs and (j,i) not in pairs:
        counter+=1
        pairs.append((i,j))
        motion = r.choice(motions2use)
        favoragainst = {"favor": models[i], "against": models[j]}
        s = time.time()
        favor_arg, against_arg, winner_json = tournament_round(models[i], models[j], judge, motion)
        e = time.time()
        # add debate data to data collector
        motions2args2winner2reasons["total_time"].append(e-s)
        motions2args2winner2reasons["judge"].append(judge)
        motions2args2winner2reasons["motions"].append(motion)
        motions2args2winner2reasons["favor_model"].append(favoragainst["favor"])
        motions2args2winner2reasons["favor_arg"].append(favor_arg)
        motions2args2winner2reasons["against_model"].append(favoragainst["against"])
        motions2args2winner2reasons["against_arg"].append(against_arg)
        virdict = chain.invoke({"message": winner_json})
        reasons = virdict.reasons
        winner = virdict.winner
        winner_model = favoragainst[winner.lower()]
        motions2args2winner2reasons["winner"].append(winner_model)
        motions2args2winner2reasons["reasons"].append(reasons)
        # add a point to the winner model 
        modelpoints[judge][winner_model] += 1
        print(f"Done with match: {judge_counter}.{counter}")
  print("Done with " + judge + " being a judge")
```

The collected data were manually annotated ([_Code reference_]()), saved to a CSV file and uploaded as [a dataset on HuggingFace hub](https://huggingface.co/datasets/as-cle-bert/DebateLLMs). 


### Post-Tournament Analysis

> _Code references: [DebateLLMChampionship_analysis.ipynb](https://github.com/AstraBert/DebateLLM-Championship/blob/main/DebateLLMChampionship_analysis.ipynb) and [MotionCategoriesAssociations.ipynb](https://github.com/AstraBert/DebateLLM-Championship/blob/main/MotionCategoriesAssociations.ipynb)_

Post-tournament analysis involved:

1. Analyzing words in motions and winning arguments when `QwQ-32B-Preview` was a judge
2. Repeating the same analysis at 1. with `Llama-3.3-70B-Instruct` as a judge
3. Repeating the same analysis at 1. with `Phi-3.5-mini-instruct` winning arguments 
4. Repeating the same analysis at 1. with with `HuggingFaceH4/starchat2-15b-v0.1` losing arguments 

We also carried out topic association analysis for winning arguments with `QwQ-32B-Preview` and `Llama-3.3-70B-Instruct` as judges, as well as the same analysis for `Phi-3.5-mini-instruct` winning arguments and `HuggingFaceH4/starchat2-15b-v0.1` losing arguments.

These are the general functions defined for the analysis:

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np

df_qwq = df[df["judge"] == "Qwen/QwQ-32B-Preview"]

def compare_winning_arg_w_motion(df: pd.DataFrame) -> Dict:
    """
    Analyzes the relationship between winning arguments and their motions.
    Returns a dictionary containing analysis results and statistics.
    """
    # Initialize containers for analysis
    keyword_overlap_scores = []
    winning_word_frequencies = Counter()
    motion_word_frequencies = Counter()
    favor_win_count = 0
    against_win_count = 0
    overlap_by_length = []

    # Analysis results
    results = {
        'overlap_scores': [],
        'word_frequencies': {},
        'winning_sides': {},
        'length_correlations': []
    }

    for index, row in df.iterrows():
        motion = row["motions"]
        motion_keywords = set(extract_keywords(motion))
        motion_word_frequencies.update(motion_keywords)

        # Determine winning argument
        is_favor_winning = row["winner"] == row["favor_model"]
        winning_arg = row["favor_arg"] if is_favor_winning else row["against_arg"]

        # Update win counters
        if is_favor_winning:
            favor_win_count += 1
        else:
            against_win_count += 1

        # Extract and analyze winning argument
        common_words = set(extract_most_common_words(winning_arg, len(motion_keywords)))
        winning_word_frequencies.update(common_words)

        # Calculate overlap score
        overlap = len(motion_keywords.intersection(common_words)) / len(motion_keywords)
        keyword_overlap_scores.append(overlap)

        # Record length correlation
        overlap_by_length.append((len(winning_arg.split()), overlap))

    # Store results
    results['overlap_scores'] = keyword_overlap_scores
    results['word_frequencies'] = {
        'motion': dict(motion_word_frequencies.most_common(20)),
        'winning_args': dict(winning_word_frequencies.most_common(20))
    }
    results['winning_sides'] = {
        'favor': favor_win_count,
        'against': against_win_count
    }
    results['length_correlations'] = overlap_by_length

    # Create visualizations
    create_analysis_plots(results)

    return results

def create_analysis_plots(results: Dict):
    """Creates and displays analysis visualizations."""
    # Set up the plotting area
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(15, 10))

    # 1. Overlap Score Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(results['overlap_scores'], bins=20)
    plt.title('Distribution of Keyword Overlap Scores')
    plt.xlabel('Overlap Score')
    plt.ylabel('Count')

    # 2. Winning Sides Pie Chart
    plt.subplot(2, 2, 2)
    sides = results['winning_sides']
    plt.pie([sides['favor'], sides['against']],
            labels=['Favor', 'Against'],
            autopct='%1.1f%%')
    plt.title('Distribution of Winning Sides')

    # 3. Word Frequencies Comparison
    plt.subplot(2, 2, 3)
    motion_words = list(results['word_frequencies']['motion'].keys())[:10]
    motion_freqs = [results['word_frequencies']['motion'][w] for w in motion_words]
    plt.barh(motion_words, motion_freqs)
    plt.title('Top 10 Motion Keywords')
    plt.xlabel('Frequency')

    # 4. Length vs Overlap Scatter Plot
    plt.subplot(2, 2, 4)
    lengths, overlaps = zip(*results['length_correlations'])
    plt.scatter(lengths, overlaps, alpha=0.5)
    plt.title('Argument Length vs Keyword Overlap')
    plt.xlabel('Argument Length (words)')
    plt.ylabel('Overlap Score')

    # Add trend line
    z = np.polyfit(lengths, overlaps, 1)
    p = np.poly1d(z)
    plt.plot(lengths, p(lengths), "r--", alpha=0.8)

    plt.tight_layout()
    plt.show()

# Helper functions (assuming these exist)
def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text. Implement your keyword extraction logic here."""
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    return [w for w in words if w.isalnum() and w not in stop_words]

def extract_most_common_words(text: str, n: int) -> List[str]:
    """Extract n most common words from text."""
    words = extract_keywords(text)
    return [word for word, _ in Counter(words).most_common(n)]
```
