
# 🧠 RAG-Based Loan Approval Q&A Chatbot

> A smart chatbot that understands **loan approval data** and answers natural language questions using **Retrieval-Augmented Generation (RAG)** powered by **LLMs**. Built with `LangChain`, `FAISS`, and `Streamlit`.

---

## 📁 Project Overview

This project builds an intelligent question-answering chatbot on top of structured loan approval data. It uses **semantic search** to retrieve relevant context from loan datasets and a **language model** to generate accurate answers — a powerful fusion of data analytics and NLP.

---

## 🔍 Key Features

- ✅ **RAG Pipeline** using `LangChain` + `FAISS`
- ✅ **Vector Embeddings** via `sentence-transformers`
- ✅ **Generative Answers** using Falcon RW-1B model
- ✅ **Data-Aware UI** built using `Streamlit`
- ✅ **Precomputed Indexing** for fast startup
- ✅ **Exploratory Data Analysis** (EDA) built-in

---

## 📂 Project Structure

| File Name                     | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `Test Dataset.csv`       | Cleaned and preprocessed loan approval training data                        |
| `Test Dataset.csv`        | Extra evaluation dataset (optional use)                                     |
| `RAG_Loan_Approval_Chatbot.ipynb` | Notebook with complete setup: loading, preprocessing, vector storage, RAG |
| `rag_index.pkl`              | FAISS index with embedded vectors and metadata for fast retrieval           |
| `app.py`                     | Streamlit UI for the chatbot interface                                      |
| `requirements.txt`           | Python dependencies                                                         |

---

## 🚀 How to Run the Chatbot

```bash
# Step 1: Clone the repository or download the files
# Step 2: Install required libraries
pip install -r requirements.txt

# Step 3: Run the Streamlit app
streamlit run app.py
```

---

## 💡 Example Questions You Can Ask

- 📌 *What is the average loan amount for approved applicants?*
- 📌 *Which education group sees more rejections?*
- 📌 *How does applicant income affect approval?*
- 📌 *Do married applicants have higher approval chances?*
- 📌 *How important is credit history in loan approval?*

---

## 📊 Data Insights Example

```python
import pandas as pd

df = pd.read_csv("Training Dataset.csv")
approved = df[df['Loan_Status'] == 'Y']
print("✅ Average Loan Amount (Approved):", round(approved['LoanAmount'].mean(), 2))
```

> ✅ Output: `Average Loan Amount (Approved): 146.41`

---

## 🛠️ Tech Stack

| Layer            | Tools Used                                  |
|------------------|----------------------------------------------|
| 🧠 Embedding      | `sentence-transformers`                      |
| 🔎 Retrieval      | `FAISS`, `LangChain`                         |
| 🤖 Generation     | `transformers` (Falcon RW-1B)                |
| 📊 Analysis       | `Pandas`                                     |
| 🌐 UI             | `Streamlit`                                  |

---

## 🏁 Project Context

This project was developed as **Assignment 8** during the **Data Science Internship** at **Celebal Technologies**.

> 💼 A perfect blend of NLP, vector search, data science, and real-time chat applications.


---

## 🚧 Future Enhancements

- 🔄 Add support for uploading new datasets for real-time analysis
- 📊 Integrate dynamic charts using Plotly or Matplotlib
- 🧠 Upgrade to OpenAI GPT/Mistral for advanced responses
- ☁️ Deploy on Streamlit Cloud or Hugging Face Spaces

---

## 👨‍💻 Author

**Keshav Kumar Jha**  
📧 [keshavkumarjha528@gmail.com](mailto:keshavkumarjha528@gmail.com)  
📍 Greater Noida, India  
🔗 [GitHub](https://github.com/jhakeshav25) • [LinkedIn](https://linkedin.com/in/keshav-kumar-jha-aa560022a/) • [LeetCode](https://leetcode.com/u/jhakeshav25/) • [GFG](https://www.geeksforgeeks.org/user/jhakeshav25/)

---

✨ *If you liked this project, give it a ⭐ and feel free to fork or contribute!*

