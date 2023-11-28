# C.E.L.T: The Sentimental Analyser
## Software Engineering Project for CSC 510
The link to **Project 3 Rubrics scorecard** README.md is [here](https://github.com/amit-99/SE_Project3/blob/master/proj3/README.md)

---

[![DOI](https://zenodo.org/badge/295188611.svg)](https://zenodo.org/badge/latestdoi/295188611)
[![GitHub Release](https://img.shields.io/github/release/amit-99/SE_Project3)](https://github.com/amit-99/SE_Project3/releases)
![Build](https://github.com/amit-99/SE_Project3/actions/workflows/main.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-v3.11+-brightgreen.svg)  
![GitHub contributors](https://img.shields.io/github/contributors/amit-99/SE_Project3)
![GitHub issues](https://img.shields.io/github/issues/amit-99/SE_Project3)
![GitHub closed issues](https://img.shields.io/github/issues-closed/amit-99/SE_Project3)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/amit-99/SE_Project3)](https://github.com/amit-99/SE_Project3)  
![GitHub language count](https://img.shields.io/github/languages/count/amit-99/SE_Project3)
![Lines of code](https://tokei.rs/b1/github/amit-99/SE_Project3)
[![GitHub-size](https://img.shields.io/github/languages/code-size/amit-99/SE_Project3)](https://github.com/amit-99/SE_Project3)
[![codecov](https://codecov.io/gh/lyonva/ClassMateBot/branch/master/graph/badge.svg)](https://app.codecov.io/gh/amit-99/SE_Project2)
[![codecov](https://codecov.io/gh/usmanwardag/dollar_bot/branch/main/graph/badge.svg?token=PYAWX95R67)](https://codecov.io/gh/usmanwardag/dollar_bot)

---
C.E.L.T. is a versatile tool that can perform sentiment analysis on different types of data, including text, audio, reviews, and news articles. Sentiment analysis is one of the fastest-growing research areas in computer science, making it challenging to keep track of all the activities in the area. In our project, we aim to achieve our goal of accurately predicting a user's sentiment by analyzing the data provided using different types of input data.

[![sentiment](https://github.com/amit-99/SE_Project2/blob/master/sentimental_analysis/realworld/static/images/sentiment_3.png)](https://youtu.be/FMuVNTV_j8U)

---

## Table of Contents
1. [Introduction](#intro)
2. [C.E.L.T. Features](#feat)
3. [How to use C.E.L.T.?](#exec)
4. [Roadmap and Progress](#roadmap)
5. [Case Study](#casestudy)
6. [Contributing to the product](#contribute)
7. [Connect with us](#Connectwithus)
8. [Team Members](#team)

---
<a name="intro"></a>
## Introduction

### What is Sentimental Analysis?
Sentiment analysis, also known as opinion mining, is the process of determining the sentiment or emotional tone in a piece of text, audio, or other forms of data. It involves identifying whether the sentiment expressed is positive, negative, or neutral.

### Why is it important?
<ul>
  <li>Sentiment analysis can help businesses and organizations understand how their customers or users feel about their products, services, or experiences. </li>
  <li>Companies can gauge public opinion about their products or services, track trends, and identify emerging issues or opportunities in the market.</li>
  <li>News agencies and media companies use sentiment analysis to analyze public sentiment towards news articles or events. This helps in generating content that aligns with the interests of the audience.</li>
  <li>Sentiment analysis is used in politics to understand public sentiment towards political candidates, parties, or policies. It is also used to gauge public opinion on social issues.</li>
</ul>

### Why use C.E.L.T.?

C.E.L.T. provides the following:
<ul>
  <li><b>Comprehensive Insights: </b>Different types of data sources provide diverse perspectives. An all-encompassing tool can provide a more comprehensive understanding of public sentiment.</li>
  <li><b>Multichannel Data Analysis:</b> In today's world, opinions and sentiments are expressed across various channels, including social media, customer reviews, audio recordings, and news articles. A tool that can analyze these diverse data sources offers a more accurate picture of public sentiment.</li>
  <li><b>Cost-Efficiency:</b> Instead of using multiple specialized tools, a single tool that can handle multiple data types is cost-effective and streamlines the analysis process.</li>
</ul>

![meme](https://github.com/amit-99/SE_Project3/blob/master/assets/images/meme.jpg?raw=true)

### How was C.E.L.T. developed?
The complete development was achieved using the following technologies:
- Python3
- Django
- HTML
- CSS
- Scrapy
- Vader Analysis Tool

Although HTML and CSS are used for the front end, the users can merge the backend logic with any of the front end frameworks they wish to use such as React, and AngularJS.

---

<a name="feat"></a>
## What can C.E.L.T. do?
|Feature|Description  |
|--|--|
|Product Analysis |```Sentimental analysis of Amazon product reviews```|
|News Analysis  |```Sentimental analysis of any recent news topic```|
|Text Analysis | ```Sentimental analysis of text input```|
|Audio Analysis   |``` Sentimental analysis of audio file``` |
|File Analysis   |``` Sentimental analysis of text file``` |
|Live Sentimental Analysis   |``` Sentimental analysis of live recorded audio``` |
|Image Sentimental Analysis   |``` Sentimental analysis of any Image file``` |

---

<a name="exec"></a>
## How to use C.E.L.T.?
### Installation
1. Clone this project:
```
git clone https://github.com/amit-99/SE_Project3.git
```
2. Make sure you are using Python 3.11 or higher. You can get it here:https://www.python.org/downloads/release/python-3115/
3. Install dependencies for the project from the root directory of the project:
```
cd <your_download_dir>\SE_Project3\
pip install -r requirements.txt
```
4. Install ffmpeg:  

For Windows:

```
winget install ffmpeg
```

For Linux (Ubuntu):  

```
sudo apt install ffmpeg
```

For Mac:  

```
brew install ffmpeg
```   
5. Run Django Server using manage.py (Note: Make sure you are in root directory of the project.)
```
cd <your_download_dir>\SE_Project3\
python .\sentimental_analysis\manage.py runserver
```
6. Next, open your browser and type in `localhost:8000` in the search bar to open the user interface of the application.

### Usage

<a name="usecases"></a>
- Start the Django server to get to the homepage
![First](https://github.com/amit-99/SE_Project3/blob/master/assets/gifs/SEProj3_1.gif)

- Amazon Product Sentiment Analysis
![second](https://github.com/amit-99/SE_Project3/blob/master/assets/gifs/SEProj3_2.gif)

- News Sentiment Analysis
![third](https://github.com/amit-99/SE_Project3/blob/master/assets/gifs/SEProj3_3.gif)

- Image Sentiment Analysis
![third](https://github.com/amit-99/SE_Project3/blob/master/assets/gifs/SEProj3_4.gif)

---
<a name="roadmap"></a>
## Roadmap and Progress
### Past Achievement(Previous Work)
- [x] Creating C.E.L.T. Django Project/Website
- [x] Sentiment Analysis Model's Algorithm addition
- [x] Text Analysis and Document Analysis Feature inclusion
- [x] Audio Analysis Feature inclusion
- [x] Case Study done for Amazon Product Review Sentiment Analysis 
- [x] Simple Documentation, Unit tests addition
- [x] Sentiment Analysis of Images
- [x] Twitter Post Sentiment Analysis (deprecated)

### Current Achievements
☑️  Fixed Amazon Product Analysis Feature<br>
☑️  Live Sentiment Analysis Feature inclusion<br>
☑️  News Analysis Feature inclusion<br>
☑️  UI Improvement for enriching User interaction with the Application<br>
☑️  Documentation Improvement for reflecting project's value accurately<br>
☑️  Addition of Builds and Workflows for better development activities<br>
☑️  Unit Tests were written and Test Coverage was improved<br>
☑️  Public facing webpage to get feedback, Raise and handle issues, Discuss and Grow with enthusiasts and supportive community<br>  

### Future Scope
- [ ] Implement User Authentication to store the history of each User
- [ ] Recommendation System based on Product Analysis Results 
- [ ] Enhance the Product Analysis by considering the number of users rated for each Product!
- [ ] Extend the Sentiment Analysis to Facebook, Twitter, and LinkedIn Posts

---  
<a name="casestudy"></a>
## Case Study: Amazon Product Review Sentiment and Text Analysis
We have done a Case Study for our Sentiment Analysis Project. It can be found [here](https://github.com/amit-99/SE_Project3/blob/master/Case_Study.md).

---
<a name="contribute"></a>
## Eager to Contribute?
To Contribute to our application, please refer to [CONTRIBUTING.md](https://github.com/amit-99/SE_Project3/blob/master/CONTRIBUTING.md)

---
<a name="Connectwithus"></a>
## Connect with us 
- Want to share your feedback or raise any issue [click here](https://noteforms.com/forms/query-form-i4mw2s)
- Need instant help? discuss over community chat to get help from other community members
- Join our mailing list for regular updates
- Join our discord forum for updates or support or discussions. [Join Here!](https://discord.gg/39aMBbWk6W)
- Or just want to be part of our journey and get to know more about the C.E.L.T and its team

**Visit our web page to be part of [C.E.L.T. community](https://factual-squash-083.notion.site/C-E-L-T-The-Sentiment-Analyzer-c71e9ace2c684b94b75fb9414e7fee75?pvs=4)**

---

<a name="team"></a>
## Team Members

- Akash Kore
- Amit Bhujbal
- Sohamkumar Patel
- Yogesh Hasabe
