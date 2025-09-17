import os
import uuid
import base64
import markdown2
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, send_file
from ollama import chat
from collections import Counter
from fpdf import FPDF
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "charts")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def scrape_website(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(url)
        screenshot_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_screenshot.png")
        driver.save_screenshot(screenshot_path)
        content = driver.page_source
        driver.quit()
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)[:10000], screenshot_path
    except Exception as e:
        return f"Error scraping: {e}", None

def extract_main_table(text):
    headers = []
    rows = []
    inside_table = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if "|" in line and "---" not in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if not inside_table:
                headers = cells
                inside_table = True
            else:
                if len(cells) == len(headers):
                    rows.append(cells)
    return headers, rows

def get_brand_counts_from_table(headers, table_data):
    if not headers or not table_data:
        return []
    try:
        brand_index = headers.index("Brand")
    except:
        brand_index = 0
    brands = [row[brand_index].capitalize() for row in table_data if len(row) > brand_index]
    return Counter(brands).most_common()

def get_pie_data_from_counts(counts):
    total = sum(count for _, count in counts)
    if total == 0:
        return []
    return [(brand, round((count / total) * 100, 1)) for brand, count in counts]

def generate_chart_image(chart_data, chart_type, title):
    if not chart_data:
        return None, None
    labels, values = zip(*chart_data)
    filename = f"{uuid.uuid4()}_{chart_type.lower()}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    plt.figure(figsize=(9, 6))
    if chart_type == "Bar":
        bars = plt.bar(labels, values, color='skyblue')
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Brand")
        plt.ylabel("Count")
        plt.title(title)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, int(yval), ha='center', va='bottom')
    elif chart_type == "Pie":
        plt.pie(values, labels=[f"{l} ({v}%)" for l, v in chart_data], autopct='%1.1f%%', startangle=140)
        plt.axis("equal")
        plt.title(title)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode(), filepath

def export_table_pdf(headers, table_data, filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    col_width = 190 / len(headers) if headers else 40
    pdf.cell(200, 10, txt="Table Data", ln=True, align='C')
    pdf.ln(10)
    if headers:
        for header in headers:
            pdf.cell(col_width, 10, txt=header, border=1)
        pdf.ln()
    for row in table_data:
        for cell in row:
            text = str(cell)
            pdf.cell(col_width, 10, txt=text.encode('latin-1', 'replace').decode('latin-1'), border=1)
        pdf.ln()
    pdf.output(filepath)
    return filename

def export_combined_pdf(summary_text, headers, table_data, bar_path, pie_path, screenshot_path, filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Web Scraper Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, summary_text)
    pdf.ln(5)
    if headers and table_data:
        col_width = 190 / len(headers) if headers else 40
        for header in headers:
            pdf.cell(col_width, 10, txt=header, border=1)
        pdf.ln()
        for row in table_data:
            for cell in row:
                text = str(cell)
                pdf.cell(col_width, 10, txt=text.encode('latin-1', 'replace').decode('latin-1'), border=1)
            pdf.ln()
    if bar_path:
        pdf.add_page()
        pdf.image(bar_path, x=10, y=30, w=180)
    if pie_path:
        pdf.add_page()
        pdf.image(pie_path, x=10, y=30, w=180)
    if screenshot_path:
        pdf.add_page()
        pdf.image(screenshot_path, x=10, y=30, w=180)
    pdf.output(filepath)
    return filename

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    summary_text = ""
    table_headers = []
    table_data = []
    bar_chart_img = pie_chart_img = ""
    bar_path = pie_path = ""
    screenshot_path = ""
    table_pdf = bar_pdf = pie_pdf = combined_pdf = ""

    if request.method == "POST":
        url = request.form.get("url")
        prompt = request.form.get("prompt")
        content, screenshot_path = scrape_website(url)

        ai_prompt = f"""You are an AI assistant that extracts structured data from web content.

TASKS:
1. Provide a short summary.
2. Output a markdown table with appropriate headers like Brand, Price, etc.
3. Provide brand counts (Apple: 3, Samsung: 4) for bar chart.
4. Provide brand-wise market share (%) for pie chart.

CONTENT:
{content}
PROMPT:
{prompt}
"""

        try:
            response = chat(model="mistral", messages=[{"role": "user", "content": ai_prompt}])
            result = response['message']['content']
            summary_text = result.split("\n\n")[0]
            table_headers, table_data = extract_main_table(result)

            bar_data = get_brand_counts_from_table(table_headers, table_data)
            pie_data = get_pie_data_from_counts(bar_data)

            bar_chart_img, bar_path = generate_chart_image(bar_data, "Bar", "Brand Frequency")
            pie_chart_img, pie_path = generate_chart_image(pie_data, "Pie", "Brand Market Share")

            table_pdf = export_table_pdf(table_headers, table_data, f"{uuid.uuid4()}_table.pdf")
            combined_pdf = export_combined_pdf(summary_text, table_headers, table_data, bar_path, pie_path, screenshot_path, f"{uuid.uuid4()}_report.pdf")

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html",
                           result=result,
                           table_headers=table_headers,
                           table_data=table_data,
                           bar_chart_img=bar_chart_img,
                           pie_chart_img=pie_chart_img,
                           table_pdf=table_pdf,
                           bar_pdf=os.path.basename(bar_path) if bar_path else None,
                           pie_pdf=os.path.basename(pie_path) if pie_path else None,
                           combined_pdf=os.path.basename(combined_pdf) if combined_pdf else None)

@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)
