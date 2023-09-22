import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

text=["The Version Control with Git course provides you with a solid, hands-on foundation for understanding the Git version control system. Git is open source software originally created by Linus Torvalds. Git manages team files for large and small projects. This allows the team to continuously improve its product. It is used by most major technology companies, and is assumed knowledge for many modern programming and IT jobs. It is a core component of DevOps, continuous delivery pipelines and cloud-native computing. You could also use Git to manage the many continuously improving revisions of that book that you are writing. In this course, you will not learn everything there is to know about Git, but you will build a strong conceptual understanding of the technology, and afterward will be able to confidently dig deeper on any topic that interests you. This course assumes no previous knowledge of Git, but if you do have experience with it, you may find this course to be both useful and challenging. This is especially true if you currently \"know just enough Git to be dangerous\". There are two paths in this course. Both rely heavily on hands-on labs. One path assumes that you have experience using a command line interface, and the other path uses the Sourcetree graphical client. If you are not experienced with a command line, we highly suggest that you go through the Sourcetree path. Eventually, you might want to go through both paths, so that you can decide which tool to use for specific tasks. Besides, repetition is good for learning :) You can watch the videos and take the quizzes from your phone if you want, but the hands-on labs require you to have a Windows or Mac computer. If you use the command line path through the course, you can also use Linux. This course uses Bitbucket (bitbucket.org) as the hosted provider for remote Git repositories. Bitbucket is free for teams of up to 5 people, including private repositories. However, most of the knowledge that you gain in this course applies to Git itself, and you can apply this knowledge to other hosted Git providers (such as GitHub). This course tries to be as concise as possible. It will probably take you about 5-10 hours to go through one of the two paths, but your mileage may vary.",
      "DevOps is the combination of cultural philosophies, practices, and tools that increases an organizations ability to deliver applications and services at high velocity: evolving and improving products at a faster pace than organizations using traditional software development and infrastructure management processes. This speed enables organizations to better serve their customers and compete more effectively in the market. DevOps process can be visualized as an infinite loop, comprising these steps: plan, code, build, test, release, deploy, operate, monitor. Throughout each phase, teams collaborate and communicate to maintain alignment, velocity, and quality. This course in the DevOps on AWS specialization focuses on code, build and test parts of the workflow. We will discuss topics such as source control, best practices for Continuous Integration, and how to use the right tools to measure code quality, by identifying workflow steps that could be automated.",
      "Welcome to the course Oracle Cloud Infrastructure Architect Professional. This course prepares you for the highest level of OCI certification, Oracle Cloud Infrastructure Architect Professional. Take a deep dive into designing and deploying Oracle Cloud Infrastructure solutions and learn Cloud-Native, microservices and serverless architectures.",
      "This introductory course is for anyone who wants a deeper dive into AWS migration. Whether you want to understand what services are helpful, need to plan a migration for your organization, or are helping other groups with their own migration, you will find valuable information throughout this course. The course sessions structure cloud migration through the three-phase migration process from AWS: assess, mobilize, and migrate and modernize. This process is designed to help your organization approach and implement a migration of tens, hundreds, or thousands of applications. By learning about this three-phase structure and the various AWS tools, features, and services that can help you during each phase you will complete this course with a better understanding of how to design and implement migrations to AWS.",
      "By completing this final capstone project you will apply various Data Analytics skills and techniques that you have learned as part of the previous courses in the IBM Data Analyst Professional Certificate. You will assume the role of an Associate Data Analyst who has recently joined the organization and be presented with a business challenge that requires data analysis to be performed on real-world datasets. You will perform the various tasks that professional data analysts do as part of their jobs, including: - Data collection from multiple sources - Data wrangling and data preparation - Exploratory data analysis - Statistical analysis and data mining - Data visualization with different charts and plots, and - Interactive dashboard creation. The project will culminate with a presentation of your data analysis report for various stakeholders in the organization. The report will include an executive summary, your analysis, and a conclusion. You will be assessed on both your work for the various stages in the Data Analysis process, as well as the final deliverable. As part of this project you will demonstrate your proficiency with using Jupyter Notebooks, SQL, Relational Databases (RDBMS), Business Intelligence (BI) tools like Cognos, and Python Libraries such as Pandas, Numpy, Scikit-learn, Scipy, Matplotlib, Seaborn and others. This project is a great addition to your portfolio and an opportunity to showcase your Data Analytics skills to prospective employers.",
      "In the second course of the Computer Vision for Engineering and Science specialization, you will perform two of the most common computer vision tasks: classifying images and detecting objects. You will apply the entire machine learning workflow, from preparing your data to evaluating your results. By the end of this course, you ll train machine learning models to classify images of street signs and detect material defects. You will use MATLAB throughout this course. MATLAB is the go-to choice for millions of people working in engineering and science, and provides the capabilities you need to accomplish your computer vision tasks. You will be provided free access to MATLAB for the course duration to complete your work. To be successful in this specialization, it will help to have some prior image processing experience. If you are new to image data, it s recommended to first complete the Image Processing for Engineering and Science specialization.",
      "This is the first of seven courses in the Google Advanced Data Analytics Certificate, which will help develop the skills needed to apply for more advanced data professional roles, such as an entry-level data scientist or advanced-level data analyst. Data professionals analyze data to help businesses make better decisions. To do this, they use powerful techniques like data storytelling, statistics, and machine learning. In this course, you ll begin your learning journey by exploring the role of data professionals in the workplace. You ll also learn about the project workflow PACE (Plan, Analyze, Construct, Execute) and how it can help you organize data projects. Google employees who currently work in the field will guide you through this course by providing hands-on activities that simulate relevant tasks, sharing examples from their day-to-day work, and helping you enhance your data analytics skills to prepare for your career. Learners who complete the seven courses in this program will have the skills needed to apply for data science and advanced data analytics jobs. This certificate assumes prior knowledge of foundational analytical principles, skills, and tools covered in the Google Data Analytics Certificate. By the end of this course, you will: -Describe the functions of data analytics and data science within an organization -Identify tools used by data professionals -Explore the value of data-based roles in organizations -Investigate career opportunities for a data professional -Explain a data project workflow -Develop effective communication skills",
      "Palo Alto Networks Network Security Fundamental, In this Network Security Fundamentals course you will gain an understanding of the fundamental tenants of network security and review the general concepts involved in maintaining a secure network computing environment. Upon successful completion of this course you will be able to describe general network security concepts and implement basic network security configuration techniques.",
      "Palo Alto Networks Network Security Fundamentals",]

# SVD decomposition
vectorizer = TfidfVectorizer(stop_words='english',smooth_idf=True)
# under the hood - lowercasing,removing special chars,removing stop words
input_matrix = vectorizer.fit_transform(text).todense()

print("step1")

svd_modeling= TruncatedSVD(n_components=9, algorithm='randomized', n_iter=100, random_state=122)
svd_modeling.fit(np.asarray(input_matrix))
components=svd_modeling.components_
vocab = vectorizer.get_feature_names_out()

#print(vocab)

print("step3")

topic_word_list = []
def get_topics(components):
  for i, comp in enumerate(components):
      terms_comp = zip(vocab,comp)
      sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:4]
      topic=""
      for t in sorted_terms:
            topic= topic + ' ' + t[0]
      topic_word_list.append(topic)
      #print(topic_word_list)
  return topic_word_list

print(get_topics(components))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a single figure with 2x2 grid of subplots
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
plt.subplots_adjust(wspace=0.2, hspace=0.4)

# Iterate through topics and generate word clouds
for i, ax in enumerate(axes.ravel()):
    wc = WordCloud(
        width=1000,
        height=600,
        margin=3,
        prefer_horizontal=0.7,
        scale=1,
        background_color='black',
        relative_scaling=0
    ).generate(topic_word_list[i])

    ax.imshow(wc)
    ax.set_title(f"Topic {i + 1}")
    ax.axis("off")

# Display the figure with all subplots
plt.show()

#########################################################################
# NMF Decomposition #####################################################

vectorizer = TfidfVectorizer(stop_words='english',smooth_idf=True)
# under the hood - lowercasing,removing special chars,removing stop words
input_matrix = vectorizer.fit_transform(text).todense()


from sklearn.decomposition import NMF
NMF_model = NMF(n_components=9, random_state=1)
W = NMF_model.fit_transform(np.asarray(input_matrix))
H = NMF_model.components_
vocab = vectorizer.get_feature_names_out()

#print(vocab)


topic_word_list = []
def get_topics(H):
  for i, comp in enumerate(H):
      terms_comp = zip(vocab,comp)
      sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
      topic=" "
      for t in sorted_terms:
            topic= topic + ' ' + t[0]
      topic_word_list.append(topic)
      #print(topic_word_list)
  return topic_word_list

print(get_topics(H))

# Create a single figure with 2x2 grid of subplots
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
plt.subplots_adjust(wspace=0.2, hspace=0.4)

# Iterate through topics and generate word clouds
for i, ax in enumerate(axes.ravel()):
    wc = WordCloud(
        width=1000,
        height=600,
        margin=3,
        prefer_horizontal=0.7,
        scale=1,
        background_color='black',
        relative_scaling=0
    ).generate(topic_word_list[i])

    ax.imshow(wc)
    ax.set_title(f"Topic {i + 1}")
    ax.axis("off")

# Display the figure with all subplots
plt.show()