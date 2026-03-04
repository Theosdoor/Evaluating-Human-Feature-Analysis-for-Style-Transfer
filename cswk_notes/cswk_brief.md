# Coursework Brief

## Content and Skills Covered

- Have a strong understanding of how to work with images and video, and to usefully manipulate visual data
- Have a good understanding of advanced deep learning models for classifying and understanding images
- Train and test generative deep learning models using real-world data
- Effective written communication
- Planning, organising and time management
- Problem solving and analysis

## General Requirements

Students are expected to work on the coursework individually.

### Expected Deliverables

Students are expected to:

1. Implement human feature extraction and style translation networks
2. Understand the limitations of existing techniques, especially with unconventional or limited datasets, and the importance of appropriate image augmentation and algorithm selection
3. Understand how deep learning models might fit back into a real-world pipeline

## Submission

> **⚠️ Please follow the instructions very carefully to avoid any score penalty.**

You should submit your report in PDF, along with a ZIP file containing:
- The Jupyter notebook that can replicate (within reason) the multimedia files as requested in the questions
- Optionally additional `.py` and `.sh` files

### Important Notes

- To maintain a small file size, apply proper image and video compression to the multimedia files
- Do **not** re-submit the original `.mp4` data files
- If you use other datasets or models, include lines in the code that download them automatically (e.g., `!git clone xxx` or `!wget xxx`)
- If your Jupyter notebook becomes too large (e.g., 50+ cells, 500+ lines of code, or 10+ distinct class functions), move portions into separate, importable Python files
- Ensure the main Jupyter notebook remains in the root directory and imports the necessary functions locally

## Assignment Context

While the techniques of computer graphics used in games have improved significantly in recent years, there are still noticeable differences between games and real-world video. It has been proposed to utilise real-world video to improve the graphics quality of games. In particular, here, we wish to apply the video style of older movies to game footage.

In this assignment, you will implement a deep learning solution to enhance the visual quality of human beings in game videos using some older movies. While it is possible to translate image styles using the whole image, the results would not be good as foreground image styles (i.e., human beings) are mixed up with background styles. With the focus being humans, it is preferable to extract the pixels that are relevant to humans using different human features.

A dataset is available on Blackboard for this assignment, which consists of `.mp4` video files from older movies and from a video game.

## Assignment Questions

### Section 1: Human Feature Analysis

#### 1.1 Human Patch Extraction (10%)

For both the game and movie videos under the "Train" folder, adapt a deep learning method to detect individual humans and extract the human image patches.

You will have much more data than what you can process – therefore, propose and develop a lightweight algorithm to gather the more useful frames.

- Store each patch as one image file
- Gather 1,000 (or more) image files of human patches
- Randomly sample and submit 50 image files
- Explain, justify and evaluate the method you adapted

**Max words: 100**

#### 1.2 Classification (15%)

Using the images created in 1.1, propose and develop a method that classifies each image into one of the five classes:

1. Full-body front view
2. Full-body back view
3. Head-and-shoulder front view
4. Head-and-shoulder back view
5. Others

Note: You may need to define what each class includes in your system.

- Randomly sample and submit 20 image files per class
- Explain, justify and evaluate your method

**Max words: 100**

#### 1.3 Training Data Selection (15%)

From the images created in 1.1, propose and develop a method that selects the most useful images for human style transfer training (in the next stage).

- You are encouraged to present and utilise your own observations and insights
- You may adapt algorithms from past questions, but note that simply doing so may not generate good enough outcomes
- Randomly sample and submit 50 image files
- Explain and justify your method

**Max words: 100**

### Section 2: Real-World Application

#### 2.1 Image Model Deployment (20%)

Implement any unpaired image-to-image network (e.g., CycleGAN) for converting images between the game and movie domains.

- Download a pretrained model or train from the beginning
- Apply it to your dataset of frames
- Analyse its performance in both directions using appropriate metrics
- Compare successful and failure cases using at least 10 images for each
- Display results in the report

**Max words: 250**

#### 2.2 Local (Temporal) Enhancement (30%)

Using your 2.1 model, transfer the style of humans in the video under the "Test" folder and submit the video.

- Critically evaluate the result and suggest possible causes of failure
- Design an improved model using local methods from 1.1 to 1.3 or other advanced temporal approaches
- Create and submit a new video
- Display at least 10 images of key frames to compare and show the improvement

**Max words: 250**

### Section 3: Report Quality (10%)

Marks for good scientific report writing: clarity, brevity, precision, and good use of diagrams/tables/visualisations.

## Reports and Guidelines

### Report Format

The report format should follow the instructions given in the questions. In all your answers, we are expecting to see:

- Enough detail (e.g., maths, analysis of results, visualisations, references to papers or other materials) to demonstrate your understanding of the subject
- Diagrams, figures, and tables to demonstrate the results and analysis
- References where suitable to support and justify your solution

### Important Notes on Implementation

The coursework aims to evaluate knowledge and understanding of both the fundamentals and the recent advances in computer vision, and not the student's programming skill. Therefore, you are free to implement solutions using any Python libraries you are most comfortable with.

You are free to re-use any pre-trained model, pre-written library/implementation or extra datasets as you see fit, **as long as attribution is clearly given in both the code and the report**. However, simply lifting existing codebases without adapting them to the problem at hand, or otherwise demonstrating an understanding of how they work, will not result in high marks.

## Word Limit Policy

The word count for each question will:

### Included in Word Count

- All the text, including in-text citations, quotations, footnotes and any other item not specifically excluded below

### Excluded from Word Count

- Diagrams, tables (including tables/lists of contents and figures)
- Equations
- Executive summary/abstract
- Acknowledgements
- Declaration
- Bibliography/list of references
- Appendices

> **Note:** It is not appropriate to use diagrams or tables merely as a way of circumventing the word limit. If a student uses a table or figure as a means of presenting their own words, then this is included in the word count.

Examiners will stop reading once the word limit has been reached, and work beyond this point will not be assessed. Checks of word counts may be carried out on submitted work, both manually and/or with the aid of the word count provided via electronic submission.

## Plagiarism and Collusion

Your assignment will be put through the plagiarism detection service.

Students suspected of plagiarism, either of published work or work from unpublished sources, will be dealt with according to Computer Science Department and University guidelines.

## Frequently Asked Questions (FAQ)

> **Be sure to read the FAQ section before asking a question.**

### Q: "Can you tell me exactly what method I should implement in doing Question X?"

**A:** Unfortunately no. One of the core parts of this assignment is for you to propose/design your solution based on the knowledge you obtained within and outside this module. This is very similar to a real-world working environment, where you need to suggest and propose solutions to your supervisor when working on a project. Telling you what methods to implement would deflect this purpose and this part of the evaluation.

### Q: "If I do X, what grades will I get?"

**A:** We cannot "pre-grade" your assignment since this will be unfair to other students who do not share this piece of information. However, we do have a formative feedback session where we can tell you how you can improve your solutions; please make sure you attend that. Also, please read this assignment brief very carefully to understand what the assignment requires.

### Q: "I've spent a million hours training these models and I still need more time. This assignment is impossible!"

**A:** This is often a problem with deep learning. Some models may take a long time to train; so you need to choose your experiments carefully. These problems are intentionally difficult; you are not expected to produce near-perfect outputs. You'll never have enough computing power to try everything; well-reasoned, well-explained models with poor results can still get good marks.

### Q: "This would be far easier if I had an expensive GPU machine!"

**A:** You are not being examined on how much GPU power you've bought. Please be explicit in your reports about the hardware you used to train your model, and the time it took. Better performing models will not score higher marks where this increase in performance is judged to have come solely from better hardware (e.g., because they have been trained for many more epochs, or because you were able to test many different hyperparameter combinations). Equally, everyone has access to the same basic resources: so waiting until just before the submission deadline to start training your models is a very bad idea!

### Q: "I found code online which looks similar to what I need. Can I use it?"

**A:** Yes, but you must cite the code in both the written report and in the comments at the top of the code. I will then carefully cross-reference this code with your implementation to see how well you have adapted their code. If you have simply copied and pasted without evidence of experimentation or tailoring, do not expect to get a good grade even if your results are good. Whereas if I see evidence of original interpretation, novel comprehension and application of the theory in the lectures, even if the experimental results are not as strong, you can get very high marks. However, if you pass off other people's code as your own, and "forget" to cite their code, or work together with other students, you will very likely get caught (see the submission Plagiarism and Collusion section on Ultra to read about the tools used to detect this). This incurs a very severe departmental penalty.

### Q: "I just read this interesting new paper which is very different to techniques from the course. Can I use it?"

**A:** Yes! You can use any papers, and even any (open-source) implementations, you like. Make sure you cite it properly in the code and reports: see the previous question.

### Q: "Can I use external datasets or even pre-trained models?"

**A:** Yes! But as with external bits of code, make sure you give proper attribution in the report and your code: and make sure you have adapted it to suit your purposes. Avoid obscenely large additional datasets, that require much greater computing power.

### Q: "Is downloading a pre-trained model an essential step to be assessed?"

**A:** No. A pre-trained model can be from irrelevant tasks, e.g., Zebra-Horse or other game-video transfer models. Training from scratch with your own design of model architecture is also fine.

### Q: "What will be the difference between using a pre-trained model or not?"

**A:** You are welcome to build up your own models. For pre-trained models, there are pre-trained movie-game models but not necessarily to be CycleGAN.

If you choose to design your own model, you are more flexible to focus on discussing the impacts of architecture, e.g., the number of layers, and adding/removing res-blocks.

If you choose to apply a pre-trained model, you can discuss the domain-shifting problem, e.g., the difference between source data distribution and the target movie-game domain distribution.

### Q: "What is the relationship between 2.1 and 2.2 and do we implement one or multiple models?"

**A:** 2.2 aims to improve 2.1 and there are two ways:

1. Using methods in 1.1 to 1.3, you can improve the same model in 2.1 by using local information
2. Developing an advanced model that can consider the temporal information in videos and get extra credit

### Q: "Many questions in section 2 are associated with that in section 1. Do we train an end-to-end model for both questions, or do we train separate models for individual questions?"

**A:** Both are fine, you have the flexibility. Apparently, building separate models for each single task is easier. Combining multiple tasks into one end-to-end model can be difficult and gives very interesting results. If you have one model that provided reasonable results, you can demonstrate a second sophisticated solution (multi-task losses of detection, classification, temporal and style transfer) to get extra credit even though the results may not be perfect.
