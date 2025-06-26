from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import requests
import ollama
import re
import matplotlib.pyplot as plt
import os
import uuid
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/charts'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

stop_words = set(stopwords.words('english'))

def scrape_website(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Error scraping website: {e}"

def basic_topic_modeling(text, n_topics=3):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf = vectorizer.fit_transform([text])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf)

        topics = []
        for topic in lda.components_:
            words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
            topics.append(words)

        return topics
    except Exception as e:
        return [[f"Error: {e}"]]

def generate_ai_response(model, content, prompt, topics):
    topic_summary = "\n".join(["Topic {}: {}".format(i+1, ", ".join(words)) for i, words in enumerate(topics)])
    full_prompt = f"""
Analyze the following website content and respond to the user's prompt.

Content:
{content[:2000]}

Identified Topics:
{topic_summary}

Prompt: {prompt}

Format your response in markdown and include:
- A summary text
- Markdown table using pipe format (|col1|col2| etc.)
- BAR: Title: My Bar Chart\nLabels: A, B, C\nValues: 10, 20, 30
- PIE: Title: My Pie Chart\nLabels: X, Y, Z\nValues: 25, 35, 40
"""
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": full_prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error from Ollama: {e}"

def save_chart(fig):
    filename = f"{app.config['UPLOAD_FOLDER']}/chart_{uuid.uuid4().hex}.png"
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename

def parse_and_generate_charts(ai_response):
    charts = []
    tables_html = []

    markdown_tables = re.findall(r"(\|.*?\|\n(?:\|.*?\|\n)+)", ai_response)
    for table in markdown_tables:
        rows = [row.strip() for row in table.strip().split("\n") if row.strip()]
        if len(rows) < 2:
            continue
        headers = [h.strip() for h in rows[0].split('|') if h.strip()]
        html = '<table class="styled-table"><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>'
        for row in rows[2:]:
            cols = [c.strip() for c in row.split('|') if c.strip()]
            html += '<tr>' + ''.join(f'<td>{c}</td>' for c in cols) + '</tr>'
        html += '</tbody></table>'
        tables_html.append(html)

    bar_match = re.search(r"BAR:.*?Title:\s*(.*?)\s*Labels:\s*(.*?)\s*Values:\s*(.*?)($|\n)", ai_response, re.DOTALL)
    if bar_match:
        title = bar_match.group(1).strip()
        labels = [x.strip() for x in bar_match.group(2).split(',')]
        values = [int(re.findall(r'\d+', x)[0]) for x in bar_match.group(3).split(',') if re.findall(r'\d+', x)]
        if len(labels) == len(values) and len(values) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.get_cmap('tab20c')(range(len(labels)))
            ax.bar(labels, values, color=colors)
            ax.set_title(title)
            ax.set_xlabel("Categories")
            ax.set_ylabel("Values")
            charts.append(save_chart(fig))

    pie_match = re.search(r"PIE:.*?Title:\s*(.*?)\s*Labels:\s*(.*?)\s*Values:\s*(.*?)($|\n)", ai_response, re.DOTALL)
    if pie_match:
        title = pie_match.group(1).strip()
        labels = [x.strip() for x in pie_match.group(2).split(',')]
        values = [int(re.findall(r'\d+', x)[0]) for x in pie_match.group(3).split(',') if re.findall(r'\d+', x)]
        if len(labels) == len(values) and len(values) > 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            ax.set_title(title)
            charts.append(save_chart(fig))

    return tables_html, charts

@app.route('/', methods=['GET', 'POST'])
def index():
    content = ""
    ai_response = ""
    tables = []
    charts = []
    url = ""
    show_prompt = False

    if request.method == "POST":
        url = request.form.get("url")
        prompt = request.form.get("prompt")
        if url and not prompt:
            content = scrape_website(url)
            show_prompt = True
        elif url and prompt:
            content = scrape_website(url)
            topics = basic_topic_modeling(content)
            ai_response = generate_ai_response("mistral", content, prompt, topics)
            if ai_response:
                tables, charts = parse_and_generate_charts(ai_response)

    return render_template('index.html', url=url, content=content, ai_response=ai_response,
                           tables=tables, charts=charts, show_prompt=show_prompt)

if __name__ == '__main__':
    app.run(debug=True)
