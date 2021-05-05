# BRIDGEi2i's Automatic Headline And Sentiment Generation

Solution Repo for Mosaic'21 [PS1](./Mosaic'21_PS1.pdf).<br>
The task was to implement a Handwritten Hindi Word Recognizer with the maximum character set for the dialect.

## Salient Features

<img src="media/salient_features.png" style="background-color:White;" alt="salient features">

We briefly explain the salient features of our approach here. In [#Approach](#approach), we explain each task in detail.  

### 1) Simpler and faster models for binary classification

- Binary classification for mobile-theme identification is not a very difficult task.
- The amount of data being processed in this step is about 4 times that being processed in the other steps. This is because the ratio of mobile-themed to non-mobile themed data is about 1:3, and we only need to do the other tasks on mobile-themed data. 
- Therefore it makes sense to use simpler and faster models for this step.

### 2) Translation of all data to english for headline generation and sentiment analysis

- Headline generation is a difficult task, which yielded poor results on multilingual data.
- Translating all data to English language using an accurate model not only provides greater scope for scalability to additional languages, it even improves performance on other tasks for which we may already have superior pretrained models in English.

### 3) Regex matching for brand identification

- The set of all possible mobile brands is a modestly-sized set
- Using regex matching instead of framing it as an NER problem is much faster and often more reliable.

### 4) Using advanced models like T5 for headline generation

- We tried a lot of possible variants but T5 performed the best.

## Complete Pipeline

<img src="media/flowchart.png" style="background-color:White;" alt="workflow">

## Approach

To deploy this project run

## Authors

- [@arch-raven](https://www.github.com/arch-raven)
